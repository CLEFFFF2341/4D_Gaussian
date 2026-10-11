"""Phase 03 infrastructure. Losses and Gaussian updates remain in author train.py."""
import builtins
import json
import os
import random
import time
import types
from pathlib import Path

import numpy as np
import ctypes
import gc
import torch

from protocol_cache import TeacherCache

ROOT = Path(__file__).resolve().parents[1]

def cpu_rss():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    assert ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return counters.WorkingSetSize


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def guard_training_rgb():
    original = builtins.open
    def checked(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            value = str(file).replace('\\', '/').lower()
            if value.endswith('.png') and ('/validation/' in value or '/test/' in value):
                raise RuntimeError('Held-out RGB access in training: ' + value)
        return original(file, *args, **kwargs)
    builtins.open = checked


def load_boundary(path):
    if not path:
        return None
    value = torch.load(path, weights_only=False)
    assert value['format'] == 'phase03_frame_boundary_v1'
    return value

def configure_initial_cloud(args):
    if not args.phase03_initial_cloud:
        return
    import scene
    from scene.dataset_readers import fetchPly
    original = scene.sceneLoadTypeCallbacks['Dynerf']
    def dense_reader(*arguments, **kwargs):
        info = original(*arguments, **kwargs)
        return info._replace(point_cloud=fetchPly(args.phase03_initial_cloud),
                             ply_path=args.phase03_initial_cloud)
    scene.sceneLoadTypeCallbacks['Dynerf'] = dense_reader


def restore_boundary(gaussians, cameras, value, generator, args):
    assert value['source_path'] == args.source_path
    assert value['seed'] == args.seed
    assert value.get('initial_cloud','') == getattr(args,'phase03_initial_cloud','')
    assert value['max_frames'] == args.max_frames or (value['frame']==1 and args.max_frames>=value['max_frames'])
    gaussians.__dict__.update(value['gaussian_state'])
    gaussians.setup_functions()
    gaussians.split_generator.set_state(value['split_rng'])
    for camera, stored in zip(cameras, value['camera_state']):
        camera.__dict__.update(stored)
    cache = CachedTeachers(args, initialize_rng=False)
    split = json.loads((ROOT/'manifests'/f'splits_{Path(args.source_path).parents[1].name}.json').read_text())['scenes'][Path(args.source_path).name]
    for index, name in zip(split['flow_runtime_indices'], split['flow_cameras']):
        if not hasattr(cameras[index], 'optical_flow_sam2_mask_final'):
            cameras[index].optical_flow_sam2_mask_final = cache.cache.b0_step(name, value['frame'], 'cuda')['motion_mask']
    # Reproduce the transition performed immediately after the boundary save.
    with torch.no_grad():
        for name in gaussians.get_atts:
            previous = gaussians.get_decoded_atts[name].clone()
            gaussians.prev_atts[name] = previous.detach()
            gaussians.prev_latents[name] = gaussians.get_atts[name].detach().clone()
            gaussians.prev_atts_initial[name] = previous.detach().clone()
    generator.setstate(value['camera_rng'])
    random.setstate(value['python_rng'])
    np.random.set_state(value['numpy_rng'])
    torch.set_rng_state(value['torch_rng'])
    torch.cuda.set_rng_state_all(value['cuda_rng'])
    return value['frame'] + 1


class CachedTeachers:
    def __init__(self, args, initialize_rng=True):
        # Preserve author constructor RNG consumption and TF32 side effects.
        # Models are unloaded before the first optimization iteration.
        if initialize_rng:
            from scene.utils import get_flow_model, get_sam2_model
            raft = get_flow_model(str(ROOT/'weights/Tartan-C-T-TSKH-spring540x960-M.pth'),
                                  str(ROOT/'working/MoRGS/RAFT/config/eval/spring-M.json'))
            sam = get_sam2_model(str(ROOT/'weights/sam2.1_hiera_large.pt'), 'configs/sam2.1/sam2.1_hiera_l.yaml')
            del raft, sam
            gc.collect()
            torch.cuda.empty_cache()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.scene = Path(args.source_path).name
        protocol = Path(args.source_path).parents[1].name
        split = json.loads((ROOT / 'manifests' / f'splits_{protocol}.json').read_text())['scenes'][self.scene]
        self.cameras = split['flow_cameras']
        cache_protocol = 'dev_phase03' if protocol=='dev' else protocol
        self.cache = TeacherCache(ROOT / 'cache/teachers' / cache_protocol, self.scene, self.cameras, 'b0',
                                  max_steps=300 if protocol=='final' else 90)
        self.step = 0
        self.index = 0

    def flow(self, *unused):
        if self.index % len(self.cameras) == 0:
            self.step += 1
        camera = self.cameras[self.index % len(self.cameras)]
        self.index += 1
        pair = self.cache.b0_step(camera, self.step, 'cuda')
        return pair['flow'][None], None, pair['heatmap'][None]

    def sam(self, predictor, images, masks):
        # The final union is restored directly in the generated training adapter.
        result = []
        for camera in self.cameras:
            result.append(self.cache.b0_step(camera, self.step, 'cuda')['motion_mask'])
        return result


def boundary_save(args, gaussians, cameras, generator, frame, frame_start, iterations, loss):
    torch.cuda.synchronize()
    save_start = time.perf_counter()
    output = Path(args.model_path)
    with torch.no_grad():
        decoded = {name: value.detach().cpu() for name, value in gaussians.get_decoded_atts.items()}
        snapshot = dict(format='phase03_decoded_render_v1', frame=frame,
                        active_sh_degree=gaussians.active_sh_degree, max_sh_degree=gaussians.max_sh_degree,
                        attributes=decoded,
                        residual=None if frame == 1 else gaussians.get_resi.detach().cpu())
        atomic_save(snapshot, output / 'snapshots' / f'{frame:04d}.pt')
    excluded = {'split_generator'}
    state = {key: value for key, value in gaussians.__dict__.items()
             if key not in excluded and not isinstance(value, (types.FunctionType, types.BuiltinFunctionType))}
    camera_state = [{key: value for key, value in camera.__dict__.items()
                     if key in ('prev_rendered', 'colors_precomp', 'cov2d_precomp', 'pix_thresh_vals',
                                'err_scale', 'mask', 'orig_mask', 'dynamic_mask', 'orig_dynamic_mask',
                                'optical_flow', 'optical_flow_mask', 'heatmap', 'optical_flow_sam2_mask',
                                'optical_flow_sam2_mask_final')}
                    for camera in cameras]
    checkpoint = dict(format='phase03_frame_boundary_v1', frame=frame, max_frames=args.max_frames,
                      initial_cloud=getattr(args,'phase03_initial_cloud',''),
                      source_path=args.source_path, seed=args.seed, gaussian_state=state,
                      camera_state=camera_state, split_rng=gaussians.split_generator.get_state(),
                      camera_rng=generator.getstate(), python_rng=random.getstate(),
                      numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all())
    atomic_save(checkpoint, output / 'checkpoints' / 'latest.pt')
    if frame == 1 or frame in (2, 5, 10) or frame % 30 == 0 or frame == args.max_frames:
        atomic_save(checkpoint, output / 'checkpoints' / f'frame_{frame:04d}.pt')
    seconds = time.perf_counter() - frame_start
    record = dict(frame=int(frame), iterations=int(iterations), points=int(gaussians.get_xyz.shape[0]),
                  loss=float(loss), train_and_save_seconds=seconds,
                  checkpoint_io_seconds=time.perf_counter()-save_start,
                  frame_setup_training_diagnostics_seconds=save_start-frame_start,
                  peak_allocated=torch.cuda.max_memory_allocated(),
                  peak_reserved=torch.cuda.max_memory_reserved(),
                  cpu_rss=cpu_rss(),
                  checkpoint_bytes=(output / 'checkpoints/latest.pt').stat().st_size)
    with (output / 'resources.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')
    print('PHASE03_FRAME ' + json.dumps(record), flush=True)
    if getattr(args, 'stop_after_frame', 0) == frame:
        print('PHASE03_CONTROLLED_STOP boundary checkpoint saved', flush=True)
        raise SystemExit(0)

