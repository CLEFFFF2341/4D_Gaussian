"""Two tiny model-level optimizer steps, without scene training or data loading."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'working/MoRGS'))
import numpy as np
import torch
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import BasicPointCloud,getProjectionMatrix
from gaussian_renderer import render_mask
from arguments import OptimizationParams

torch.manual_seed(0)
started=time.perf_counter()
torch.cuda.reset_peak_memory_stats()
config=json.loads((ROOT/'manifests/resolved_config.json').read_text(encoding='utf-8'))
groups={k:SimpleNamespace(**v) for k,v in config['groups'].items()}
qp=groups['quantize_params']
qp.use_shift=[bool(x) for x in qp.use_shift]
qp.seed=0
model=GaussianModel(groups['model_params'].sh_degree,qp,groups['model_params'])
rng=np.random.default_rng(0)
points=rng.uniform(-.5,.5,(32,3)).astype(np.float32)
points[:,2]+=6
pcd=BasicPointCloud(points=points,colors=rng.uniform(.1,.9,(32,3)).astype(np.float32),
                    normals=np.zeros_like(points))
model.create_from_pcd(pcd,1.)
camera=MiniCam(uid=0,width=64,height=48,fovy=np.pi/2,fovx=np.pi/2,znear=.1,zfar=100.,
    world_view_transform=torch.eye(4,device='cuda'),
    full_proj_transform=getProjectionMatrix(.1,100.,np.pi/2,np.pi/2).T.contiguous().cuda())
pipe=groups['pipe_params']
records=[]
opt=OptimizationParams(groups['opt_params_initial'],groups['opt_params_rest'])
for frame in [1,2]:
    if frame==2:
        with torch.no_grad():
            for name,value in model.get_atts.items():
                model.prev_atts[name]=value.detach().clone()
                model.prev_latents[name]=model._latents[name].detach().clone()
            model.frame_idx=2
            model.update_residuals()
            model._latents['xyz'][:,0].add_(.05)
    opt.set_params(frame)
    model.training_setup(opt)
    model.update_learning_rate(1,qp)
    before=model.get_xyz.detach().clone()
    out=render_mask(camera,model,pipe,torch.zeros(3,device='cuda'),
                    render_flow=True,render_motion_label=True,backward_alpha=True)
    assert out['visibility_filter'].any()
    loss=out['render'].square().mean()+.01*(out['flow']/(out['alpha']+1e-6)).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    gradients={name:value.grad.norm().item() for name,value in model._latents.items()
               if value.grad is not None}
    assert all(np.isfinite(v) for v in gradients.values())
    assert gradients['xyz']>0
    if frame==2:
        assert model.offset_3d.grad is not None and torch.isfinite(model.offset_3d.grad).all()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    change=(model.get_xyz.detach()-before).abs().max().item()
    assert change>0
    records.append(dict(frame=frame,loss=loss.item(),gradients=gradients,
                        actual_xyz_max_update=change,
                        optimizer_has_offset=any(x['name']=='offset_3d' for x in model.optimizer.param_groups)))
result=dict(status='PASS',points=32,steps=records,torch=torch.__version__,cuda=torch.version.cuda,
            seconds=time.perf_counter()-started,peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
            caveat='Synthetic two-step model/renderer/optimizer check; no scene training, densification or restart claim.')
(ROOT/'manifests/official_model_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
