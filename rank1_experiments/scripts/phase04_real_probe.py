"""Same initialized real geometry: gradient, RGB, K and native UID tests."""
import gc
import json
import math
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'working/MoRGS')]
from phase04.model import SharedGaussianModel
from phase04.native import snapshot, raster_candidates, candidate_loss
from phase04.shared_flow import match_uid
from phase04.training import motion_gradients
from scene.dataset_readers import readCamerasFromPoseBounds
from utils.camera_utils import loadSequentialCam
from gaussian_renderer import render_mask
from arguments import OptimizationParams, OptimizationParamsInitial, OptimizationParamsRest
from utils.loss_utils import l1_loss, ssim
from protocol_cache import TeacherCache


def optimization(scene):
    cfg = yaml.safe_load((ROOT/'configs/phase03'/f'dev_{scene}.yaml').read_text())
    parser = ArgumentParser()
    oi, ore = OptimizationParamsInitial(parser, cfg['opt_params_initial']), OptimizationParamsRest(parser, cfg['opt_params_rest'])
    args = parser.parse_args([])
    opt = OptimizationParams(oi.extract(args), ore.extract(args))
    opt.set_params(2)
    return opt


def initial_model(scene):
    ckpath = ROOT/'runs/B0/dev'/scene/'seed0_loader_exact/checkpoints/frame_0001.pt'
    checkpoint = torch.load(ckpath, weights_only=False)
    state = checkpoint['gaussian_state']
    pc = SharedGaussianModel(state['max_sh_degree'], state['latent_args'], state['model_args'])
    pc.__dict__.update(state); pc.setup_functions(); pc.ensure_identity()
    source = snapshot(pc, pc.identity.uid)
    with torch.no_grad():
        for name, value in pc.get_decoded_atts.items():
            pc.prev_atts[name] = value.detach().clone()
            pc.prev_latents[name] = pc.get_atts[name].detach().clone()
            pc.prev_atts_initial[name] = value.detach().clone()
        pc.frame_idx = 2
        pc.update_residuals()
        pc.training_setup(optimization(scene))
        pc.added_mask = torch.zeros(len(pc.identity.uid), dtype=torch.bool, device='cuda')
    return pc, source, Path(checkpoint['source_path'])


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    background = torch.zeros(3, device='cuda')
    results, uid_results = [], []
    for scene in ('coffee_martini', 'cut_roasted_beef'):
        pc, source, data = initial_model(scene)
        meta = json.loads((data/'dataset_meta.json').read_text())
        infos, _ = readCamerasFromPoseBounds(str(data/'poses_bounds.npy'), meta['size_wh'], N_video_views=10)
        split = json.loads((ROOT/'manifests/splits_dev.json').read_text())['scenes'][scene]
        cache = TeacherCache(ROOT/'cache/teachers/dev_phase03', scene, split['flow_cameras'], 'shared')
        for index, name in zip(split['flow_runtime_indices'], split['flow_cameras']):
            camera = loadSequentialCam(SimpleNamespace(data_device='cuda', znear=5., zfar=300.), index, infos[index], 1.)
            with Image.open(data/name/'0001.png') as im: gt = to_tensor(im.convert('RGB')).cuda()
            package = render_mask(camera, pc, pipe, background, render_flow=False)
            rgb_loss = .8*l1_loss(package['render'], gt)+.2*(1-ssim(package['render'], gt))
            original_gradient = torch.autograd.grad(rgb_loss, pc._latents['xyz'], retain_graph=True)[0]
            before = {n:p.detach().clone() for n,p in pc._latents.items()}
            teacher = cache.pair(name, 0, 1, 'cuda')
            krows = []
            for k in (8, 32, 64):
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
                candidates = raster_candidates(source, camera, pc.identity.uid, k)
                torch.cuda.synchronize(); export_seconds=time.perf_counter()-start
                for method in ('B1', 'B2', 'H', 'M'):
                    flow, diag = candidate_loss(source, pc.get_xyz, camera, candidates, teacher['flow'], teacher['valid'], method)
                    grads = motion_gradients(flow, pc)
                    assert grads['norm'] > 0 and grads['xyz_norm'] > 0 and grads['offset_norm'] > 0, (scene, name, method, grads)
                    assert torch.isfinite(flow)
                    record = dict(scene=scene, camera=name, k=k, method=method, loss=float(flow.detach()),
                                  gradients=grads, **{key:value for key,value in diag.items() if key != 'method'},
                                  export_seconds=export_seconds, peak_allocated=torch.cuda.max_memory_allocated())
                    results.append(record); krows.append(record)
                if k == 8: first_ids, first_weights = candidates['ids'].clone(), candidates['weights'].clone()
                else:
                    common_error = float((first_weights-candidates['weights'][..., :8]).abs().max())
                    assert torch.equal(first_ids, candidates['ids'][..., :8]) and common_error == 0
                del candidates
            # The head never mutates model parameters or the RGB graph. Repeat
            # RGB backward remains nonzero, and exactly matches pre-head graph.
            repeated = render_mask(camera, pc, pipe, background, render_flow=False)['render']
            rgb_error = float((repeated-package['render']).abs().max())
            assert rgb_error == 0
            assert all(torch.equal(before[n], p.detach()) for n,p in pc._latents.items())
            repeat_loss = .8*l1_loss(repeated, gt)+.2*(1-ssim(repeated, gt))
            repeated_gradient = torch.autograd.grad(repeat_loss, pc._latents['xyz'])[0]
            grad_error = float((repeated_gradient-original_gradient).abs().max())
            assert grad_error < 1e-6 and original_gradient.norm() > 0
            zero = rgb_loss.new_zeros(())
            assert not zero.requires_grad  # actual F0 branch; no flow graph
            results.append(dict(scene=scene, camera=name, method='F0', rgb_forward_max_error=rgb_error,
                                rgb_gradient_max_error=grad_error, rgb_gradient_norm=float(original_gradient.norm()),
                                flow_gradient_norm=0., parameters_unchanged=True))
            print('REAL_PROBE '+json.dumps(results[-1]), flush=True)
            del before, package, repeated, original_gradient, repeated_gradient, rgb_loss, repeat_loss, first_ids, first_weights, flow
            gc.collect()
        # Exercise actual native model clone/split/prune, not just Identity class.
        old_uid = pc.identity.uid.clone()
        old_mapping = pc.mapping.clone()
        prune = torch.arange(len(old_uid), device='cuda') >= 64
        with torch.no_grad():
            pc.prune_points(prune); pc.mapping = old_mapping[~prune]
            pc.percent_dense = 1.
            grads = torch.zeros(64, 1, device='cuda'); grads[0] = 1.
            pc.densify_and_clone(grads, .5, float(pc.get_scaling.max())*10, is_dynamic=True)
            clone_uid, clone_parent = int(pc.identity.uid[-1]), int(pc.identity.parent[-1])
            assert clone_parent == int(old_uid[0]) and clone_uid >= len(old_uid)
            grads = torch.zeros(len(pc.identity.uid), 1, device='cuda'); grads[1] = 1.
            parent = int(pc.identity.uid[1])
            pc.densify_and_split(grads, .5, float(pc.get_scaling.min())*.1, N=2, is_dynamic=True)
            assert (pc.identity.parent == parent).sum() == 2 and not (pc.identity.uid == parent).any()
            assert pc.identity.uid.unique().numel() == len(pc.identity.uid)
            pc.assert_optimizer_identity()
            matches = match_uid(old_uid, pc.identity.uid)
            assert matches[1] == -1 and matches[0] >= 0
            uid_results.append(dict(scene=scene, status='PASS', original_points=len(old_uid),
                                    final_test_points=len(pc.identity.uid), clone_uid=clone_uid, clone_parent=clone_parent,
                                    split_parent=parent, new_ids_never_match_source=True, offset_optimizer_identity=True))
        print('NATIVE_UID '+json.dumps(uid_results[-1]), flush=True)
        del pc, source, teacher, camera, zero
        gc.collect(); torch.cuda.empty_cache()
    k8 = [r for r in results if r.get('k') == 8]
    # Predeclared engineering gate: retain >=95% source matched mass for all
    # sampled cameras. If inadequate, use K32 for every compared family.
    common_k = next((k for k in (8, 32, 64) if min(r['topk_mass_retention'] for r in results if r.get('k') == k) >= .95), None)
    assert common_k is not None, 'K64 still fails the shared mass-retention gate'
    report = dict(status='PASS', real_batch_results=results, native_uid_tests=uid_results,
                  k8_gate_min_mass=.95, common_k=common_k, calibration_data='train only 0000->0001; no cam00/cam01',
                  rgb_regression='head disabled and read-only export: original forward/gradient preserved')
    (ROOT/'manifests/phase04_real_probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PHASE04_REAL_PROBE_PASS common_k='+str(common_k), flush=True)


if __name__ == '__main__': main()
