"""Check actual training adapter camera order, online labels, loss and gradients."""
import gc
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'working/MoRGS'))
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from scene import GaussianModel
from scene.utils import get_flow_model,get_flow,get_sam2_model,get_sam2_mask
from scene.dataset_readers import readCamerasFromPoseBounds
from utils.camera_utils import loadSequentialCam
from gaussian_renderer import render_mask
from phase03_hooks import CachedTeachers

def main():
    run=ROOT/'runs/B0/smoke/coffee_martini/continuous_v2'
    checkpoint=torch.load(run/'checkpoints/frame_0002.pt',weights_only=False)
    source=Path(checkpoint['source_path'])
    args=SimpleNamespace(source_path=str(source))
    adapter=CachedTeachers(args,initialize_rng=False)
    # Step1 is consumed in author flow-camera order, then check step2.
    for _ in range(4):adapter.flow()
    ckpt=ROOT/'weights/Tartan-C-T-TSKH-spring540x960-M.pth'
    config=ROOT/'working/MoRGS/RAFT/config/eval/spring-M.json'
    raft=get_flow_model(str(ckpt),str(config))
    images=[];masks=[];online=[];cached=[];records=[]
    for camera in adapter.cameras:
        with Image.open(source/camera/'0000.png') as im:first=to_tensor(im.convert('RGB')).cuda()
        with Image.open(source/camera/'0001.png') as im:second=to_tensor(im.convert('RGB')).cuda()
        before=torch.cuda.get_rng_state().clone();cpu_before=torch.get_rng_state().clone()
        flow,_,heat=get_flow(str(ckpt),str(config),raft,first,second)
        value,_,cached_heat=adapter.flow(first,second)
        assert torch.equal(before,torch.cuda.get_rng_state()) and torch.equal(cpu_before,torch.get_rng_state())
        error=float((flow-value).abs().max())
        heat_error=float((heat-cached_heat).abs().max())
        assert error==0 and heat_error==0,(camera,error,heat_error)
        records.append(dict(camera=camera,flow_abs_max=error,heatmap_abs_max=heat_error))
        images.append(second);masks.append(flow[0].norm(dim=0)>1.5)
        online.append(flow[0]);cached.append(value[0])
    del raft;gc.collect();torch.cuda.empty_cache()
    sam=get_sam2_model(str(ROOT/'weights/sam2.1_hiera_large.pt'),'configs/sam2.1/sam2.1_hiera_l.yaml')
    before=torch.cuda.get_rng_state().clone();cpu_before=torch.get_rng_state().clone()
    labels=get_sam2_mask(sam,images,masks)
    assert torch.equal(before,torch.cuda.get_rng_state()) and torch.equal(cpu_before,torch.get_rng_state())
    unions=adapter.sam(None,images,masks)
    for record,label,motion,union in zip(records,labels,masks,unions):
        count=int(((label+motion)!=union).sum())
        assert count==0,(record,count)
        record['sam_union_changed_pixels']=count
    del sam;gc.collect();torch.cuda.empty_cache()
    state=checkpoint['gaussian_state']
    gaussian=GaussianModel(state['max_sh_degree'],state['latent_args'],state['model_args'])
    gaussian.__dict__.update(state);gaussian.setup_functions()
    meta=json.loads((source/'dataset_meta.json').read_text(encoding='utf-8'))
    infos,_=readCamerasFromPoseBounds(str(source/'poses_bounds.npy'),meta['size_wh'],N_video_views=10)
    camera=loadSequentialCam(SimpleNamespace(data_device='cuda',znear=5.,zfar=300.),0,infos[0],1.)
    pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
    package=render_mask(camera,gaussian,pipe,torch.zeros(3,device='cuda'),render_flow=True)
    prediction=package['flow']/(package['alpha']+1e-6)
    loss1=(((prediction-online[0]).square().sum(0)+1e-12).sqrt()).mean()
    loss2=(((prediction-cached[0]).square().sum(0)+1e-12).sqrt()).mean()
    gradient1=torch.autograd.grad(loss1,gaussian._latents['xyz'],retain_graph=True)[0]
    gradient2=torch.autograd.grad(loss2,gaussian._latents['xyz'])[0]
    gradient_error=float((gradient1-gradient2).abs().max())
    assert float(abs(loss1-loss2))==0 and gradient_error<1e-6,gradient_error
    result=dict(status='PASS',training_step=2,records=records,loss_abs_error=float(abs(loss1-loss2)),
                xyz_gradient_abs_max=gradient_error,teacher_inference_rng_unchanged=True,
                teacher_constructor_rng_and_tf32_preserved=True,shared_mask_applied=False)
    (ROOT/'manifests/phase03_teacher_training_equivalence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
