"""Verify actual M checkpoint resume, identities, optimizer and normal variability."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'working/MoRGS')]
from phase04.model import SharedGaussianModel
from phase03_evaluate import RenderSnapshot
from scene.dataset_readers import readCamerasFromPoseBounds
from utils.camera_utils import loadSequentialCam
from gaussian_renderer import render_mask

paths = [ROOT/'runs/phase04/native/coffee_martini/M']+[
    ROOT/'runs/phase04/resume/coffee_martini'/name for name in ('resume', 'repeat')]
checkpoints = [torch.load(p/'checkpoints/latest.pt', weights_only=False) for p in paths]
snapshots = [torch.load(p/'snapshots/0004.pt', weights_only=False) for p in paths]
for checkpoint in checkpoints:
    state = checkpoint['gaussian_state']
    pc = SharedGaussianModel(state['max_sh_degree'], state['latent_args'], state['model_args'])
    pc.__dict__.update(state); pc.setup_functions(); pc.assert_optimizer_identity()
    assert len(pc.identity.uid) == len(pc._latents['xyz'])
    assert pc.identity.uid.unique().numel() == len(pc.identity.uid)
    assert pc.phase04_flow_lambda > 0
    assert checkpoint['frame'] == 4
counts = [len(c['gaussian_state']['identity'].uid) for c in checkpoints]
assert abs(counts[0]-counts[1]) <= max(16, 2*abs(counts[1]-counts[2]))
lambdas = [c['gaussian_state']['phase04_flow_lambda'] for c in checkpoints]
assert lambdas[0] == lambdas[1] == lambdas[2]
source = Path(checkpoints[0]['source_path'])
meta = json.loads((source/'dataset_meta.json').read_text())
infos, _ = readCamerasFromPoseBounds(str(source/'poses_bounds.npy'), meta['size_wh'], N_video_views=10)
models = [RenderSnapshot(s) for s in snapshots]
pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
records = []
with torch.no_grad():
    for index in (0, 3, 7, 12):
        camera = loadSequentialCam(SimpleNamespace(data_device='cuda', znear=5., zfar=300.), index, infos[index], 1.)
        images = [render_mask(camera, pc, pipe, torch.zeros(3, device='cuda'))['render'].clamp(0,1) for pc in models]
        resume_mae = float((images[0]-images[1]).abs().mean())
        repeat_mae = float((images[1]-images[2]).abs().mean())
        tolerance = max(1e-5, 2*repeat_mae)
        assert resume_mae <= tolerance and resume_mae < 1/255, (index,resume_mae,repeat_mae,tolerance)
        records.append(dict(camera=meta['cameras'][index], resume_mae=resume_mae, repeat_mae=repeat_mae,
                            tolerance=tolerance, abs_max=float((images[0]-images[1]).abs().max())))
report = dict(status='PASS', frames=[c['frame'] for c in checkpoints], points=counts,
              retained_lambda=lambdas, uid_unique=True, actual_offset_optimizer_identity=True,
              render_comparisons=records, bitwise_determinism_claimed=False)
(ROOT/'manifests/phase04_resume_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
