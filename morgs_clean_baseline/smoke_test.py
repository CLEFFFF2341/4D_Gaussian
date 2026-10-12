import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
import common,engine,torch
import json, numpy as np
from PIL import Image
from utils.loss_utils import l1_loss,ssim

rows=[]
for scene in ('coffee_martini','cut_roasted_beef'):
    cfg,model,quant,opt=common.configuration(scene,0)
    data=common.Dataset(scene)
    path=ROOT/'morgs_clean_baseline/reference'/scene/'seed0/frame_0090.pt'
    if not path.exists(): path=path.parent/'latest.pt'
    pc,ck=common.load(path,opt)
    bg=torch.zeros(3,device='cuda')
    with torch.no_grad(): out=engine.render(data.valcam,pc,bg)['render']
    saved=list((path.parent/'renders').glob('0089.png'))
    assert len(saved)==1,saved
    previous=torch.from_numpy(np.asarray(Image.open(saved[0]).convert('RGB')).copy()).cuda().permute(2,0,1)/255
    # Saved images are 8-bit exports: tolerate one quantization bin.
    error=float((out.clamp(0,1)-previous).abs().max())
    assert error<=1/255+3e-5,error
    assert pc.param_names==['xyz','f_dc','f_rest','sc','rot','op']
    assert pc.offset_3d is None and pc.gate_atts is None
    assert list(pc._latents)==pc.param_names
    assert all(not getattr(pc.model_args,k) for k in
               ('morgs_flow_supervision','morgs_motion_offset','morgs_motion_confidence'))
    opt.set_params(90)
    pc.optimizer.zero_grad(set_to_none=True)
    image=engine.render(data.cameras[0],pc,bg)['render']
    gt=data.images(90)[0]
    loss=(1-opt.lambda_dssim)*l1_loss(image,gt)+opt.lambda_dssim*(1-ssim(image,gt))
    loss.backward()
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in pc._latents.values())
    pc.optimizer.step()
    assert torch.isfinite(pc._latents['xyz']).all()
    forbidden_imports=[k for k in sys.modules if k.startswith(('n01','morgs_n01','raft','sea_raft','sam2'))]
    assert not forbidden_imports,forbidden_imports
    assert set(pc.latent_decoders)==set(pc.param_names)
    assert Path(engine.native._C.__file__).resolve().is_relative_to(HERE/'native')
    rows.append(dict(scene=scene,frame=ck['frame'],points=len(pc.get_xyz),max_export_error=error,
                     loss=float(loss),optimizer_step_finite=True,forbidden_imports=forbidden_imports))
    del pc,data,out,image,gt,loss
    torch.cuda.empty_cache()

# Exercise the actual frame-1 -> frame-2 residual transition on a tiny update.
scene='coffee_martini'
_,_,quant,opt=common.configuration(scene,0)
data=common.Dataset(scene)
pc,ck=common.load(HERE/'bootstrap'/scene/'seed0/frame_0001.pt',opt)
images=data.images(2);opt.set_params(2);opt.iterations=8*len(images)
pc.frame_idx=2;pc.origin=torch.arange(len(pc.uid),device='cuda')
with torch.no_grad():
    pc.update_residuals();pc.training_setup(opt)
before=pc.get_xyz.detach().clone()
for iteration,i in enumerate((0,1),1):
    pc.update_learning_rate(iteration,quant)
    out=engine.render(data.cameras[i],pc,bg)
    loss=(1-opt.lambda_dssim)*l1_loss(out['render'],images[i])+opt.lambda_dssim*(1-ssim(out['render'],images[i]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pc._latents.values())
    pc.accumulate(out);pc.optimizer.step();pc.optimizer.zero_grad(set_to_none=True)
pc.assert_identity();common.commit(pc)
assert pc.offset_3d is None and pc.gate_atts is None
assert list(pc._latents)==['xyz','f_dc','f_rest','sc','rot','op']
saved=HERE/'smoke_outputs/frame_0002.pt'
common.save(saved,pc,2,0,dict(purpose='two-step smoke check'))
reload,ck2=common.load(saved,opt)
with torch.no_grad():
    a=engine.render(data.valcam,pc,bg)['render']
    b=engine.render(data.valcam,reload,bg)['render']
assert torch.equal(a,b)
rows.append(dict(scene=scene,frame=2,steps=2,residual_transition=True,
                 points=len(pc.uid),position_updated=bool(not torch.equal(before,pc.get_xyz)),
                 checkpoint_roundtrip_exact=True,loss=float(loss)))
(HERE/'backbone_checks.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
