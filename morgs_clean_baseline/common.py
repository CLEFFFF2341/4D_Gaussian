import sys, json, random, copy, time, argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'rank1_experiments/working/MoRGS'))
import importlib.util
camera_spec=importlib.util.spec_from_file_location('scene.utils',HERE/'clean_scene_utils.py')
camera_module=importlib.util.module_from_spec(camera_spec)
sys.modules['scene.utils']=camera_module
camera_spec.loader.exec_module(camera_module)
import engine
from clean_model import CleanModel as AuditedModel
from arguments import ModelParams,QuantizeParams,OptimizationParams,OptimizationParamsInitial,OptimizationParamsRest
from scene.dataset_readers import fetchPly,getNerfppNorm,readCamerasFromPoseBounds
from utils.camera_utils import loadSequentialCam
def configuration(scene, seed):
    path = HERE / 'configs' / f'{scene}.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['model_params']['seed'] = seed
    cfg['quantize_params']['seed'] = seed
    p = argparse.ArgumentParser()
    mp = ModelParams(p, cfg['model_params'])
    qp = QuantizeParams(p, cfg['quantize_params'])
    oi = OptimizationParamsInitial(p, cfg['opt_params_initial'])
    ore = OptimizationParamsRest(p, cfg['opt_params_rest'])
    a = p.parse_args([])
    (model, quant) = (mp.extract(a), qp.extract(a))
    quant.use_shift = [bool(x) for x in quant.use_shift]
    quant.seed = seed
    opt = OptimizationParams(oi.extract(a), ore.extract(a))
    opt.set_params(1)
    model.morgs_flow_supervision = False
    model.morgs_motion_offset = False
    model.morgs_motion_confidence = False
    cfg['disabled_morgs_modules'] = ['flow_supervision','motion_offset','motion_confidence']
    cfg['baseline_revision'] = 'morgs_removed_three_v1'
    for key, value in vars(quant).items():
        if isinstance(value,list) and len(value)==8: setattr(quant,key,value[:6])
    cfg['source_config'] = str(path)
    for section in ('model_params','quantize_params','opt_params_initial','opt_params_rest'):
        for key in list(cfg[section]):
            if any(token in key for token in ('flow','motion_label','sam','offset_lr','lambda_offset','frz_offset')):
                del cfg[section][key]
    cfg['model_params'].update(morgs_flow_supervision=False,morgs_motion_offset=False,morgs_motion_confidence=False)
    cfg['effective_quantize_params'] = copy.deepcopy(vars(quant))
    cfg['training_loss'] = '(1-lambda_dssim)*RGB_L1 + lambda_dssim*(1-RGB_SSIM)'
    return (cfg, model, quant, opt)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class Dataset:

    def __init__(self, scene):
        self.scene = scene
        self.path = ROOT / 'rank1_experiments/data/dynerf/dev/train' / scene
        self.valpath = ROOT / 'rank1_experiments/data/dynerf/dev/validation' / scene
        self.split = json.loads((ROOT / 'rank1_experiments/manifests/splits_dev.json').read_text())['scenes'][scene]
        meta = json.loads((self.path / 'dataset_meta.json').read_text())
        (self.infos, _) = readCamerasFromPoseBounds(str(self.path / 'poses_bounds.npy'), meta['size_wh'], N_video_views=90)
        vmeta = json.loads((self.valpath / 'dataset_meta.json').read_text())
        assert vmeta['cameras'] == ['cam01']
        (vinfos, _) = readCamerasFromPoseBounds(str(self.valpath / 'poses_bounds.npy'), vmeta['size_wh'], N_video_views=90)
        self.camargs = SimpleNamespace(data_device='cuda', znear=5.0, zfar=300.0)
        self.cameras = [loadSequentialCam(self.camargs, i, info, 1.0) for (i, info) in enumerate(self.infos)]
        self.valcam = loadSequentialCam(self.camargs, 0, vinfos[0], 1.0)
        self.extent = getNerfppNorm(self.infos)['radius']

    def images(self, frame):
        out = []
        for (i, name) in enumerate(self.split['train']):
            path = self.path / name / f'{frame - 1:04d}.png'
            with Image.open(path) as im:
                out.append(to_tensor(im.convert('RGB')).cuda())
            self.cameras[i].original_image = out[-1]
            self.cameras[i].frame_idx = frame
        return out

    def gt(self, frame):
        with Image.open(self.valpath / 'cam01' / f'{frame - 1:04d}.png') as im:
            return to_tensor(im.convert('RGB')).cuda()

def commit(pc):
    with torch.no_grad():
        decoded = pc.get_decoded_atts
        for (name, value) in decoded.items():
            pc.prev_atts[name] = value.detach().clone()
            pc.prev_latents[name] = pc.get_atts[name].detach().clone()
            pc.prev_atts_initial[name] = value.detach().clone()

def save(path, pc, frame, seed, extra=None):
    state = {k: v for (k, v) in vars(pc).items() if not callable(v) and k not in ('optimizer', 'split_generator')}
    obj = dict(state=state, optimizer=pc.optimizer.state_dict(), frame=frame, seed=seed, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), split_rng=pc.split_generator.get_state(), extra=extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)

def load(path, opt, restore_optimizer=True):
    ck = torch.load(path, weights_only=False)
    assert ck['state'].get('baseline_revision') == 'morgs_removed_three_v1', 'Old MORGS initialization forbidden'
    state = ck['state']
    assert set(state['_latents'])==set(('xyz','f_dc','f_rest','sc','rot','op')), 'Six attributes required'
    pc = AuditedModel(state['max_sh_degree'], state['latent_args'], state['model_args'])
    pc.__dict__.update(state)
    pc.setup_functions()
    stats = {k: state[k] for k in ('xyz_gradient_accum', 'infl_accum', 'denom', 'infl_denom', 'added_mask')}
    pc.training_setup(opt)
    if restore_optimizer:
        pc.optimizer.load_state_dict(ck['optimizer'])
        pc.__dict__.update(stats)
    torch.set_rng_state(ck['rng'])
    torch.cuda.set_rng_state_all(ck['cuda_rng'])
    pc.split_generator.set_state(ck['split_rng'])
    pc.assert_identity()
    return (pc, ck)

def camera_order(seed, frame, count, steps):
    gen = random.Random(seed * 100003 + frame * 1009)
    (stack, out) = ([], [])
    for _ in range(steps):
        if not stack:
            stack = list(range(count))
        out.append(stack.pop(gen.randrange(len(stack))))
    return out
