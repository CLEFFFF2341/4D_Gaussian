"""Build B0-exact FP32 flow and separate corrected/shared FP16 flow caches."""
import argparse
from functools import lru_cache
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from phase02_prepare_data import ROOT, DEV, FLOW, sha, save
sha=lru_cache(maxsize=32)(sha)
WORK=ROOT/'working/MoRGS';sys.path.insert(0,str(WORK))
CKPT=ROOT/'weights/Tartan-C-T-TSKH-spring540x960-M.pth'
CFG=WORK/'RAFT/config/eval/spring-M.json'
CACHE=ROOT/'cache/teachers/dev'
def image(scene,cam,frame):
    assert cam in FLOW[scene] and cam not in ['cam00','cam01']
    p=ROOT/f'data/dynerf/dev/train/{scene}/{cam}/{frame:04d}.png'
    a=np.asarray(Image.open(p).convert('RGB')).copy()
    return torch.from_numpy(a).permute(2,0,1).float().cuda()/255
def pair_path(scene,cam,a,b):return CACHE/f'{scene}/{cam}/{a:04d}_{b:04d}.npz'
def fb_mask(f,b):
    _,_,h,w=f.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=f.device),torch.arange(w,device=f.device),indexing='ij')
    x=xx[None]+f[:,0];y=yy[None]+f[:,1]
    grid=torch.stack([2*x/(w-1)-1,2*y/(h-1)-1],-1)
    backward=F.grid_sample(b,grid,mode='bilinear',padding_mode='zeros',align_corners=True)
    # Express both vectors in reference 1352x1014 pixels before squared threshold.
    fref=f*2;bref=backward*2
    err=(fref+bref).square().sum(1)
    valid=(x>=0)&(x<=w-1)&(y>=0)&(y<=h-1)&torch.isfinite(f).all(1)&torch.isfinite(backward).all(1)
    valid &= err <= 0.01*(fref.square().sum(1)+bref.square().sum(1))+0.5
    return valid[0],err[0]
def metadata(scene,cam,a,b):
    return dict(scene=scene,camera=cam,source_frame=a,target_frame=b,shape_hw=[507,676],
      flow_units='destination pixel displacement on source image grid; x right/y down',
      reference_wh=[1352,1014],checkpoint_sha256=sha(CKPT),config_sha256=sha(CFG),
      code_sha256=sha(WORK/'scene/utils.py'),input_range='RGB float32 [0,1], upstream MoRGS behavior',
      b0_precision='float32 lossless',shared_precision='float16',
      shared_change='y*=507/506 after upstream odd-height resize; input normalization unchanged',
      b0_valid_mask='all pixels; shared_valid must NEVER be applied to original B0 loss',
      fb_threshold='Squared reference pixel units: <=0.01*(f^2+warped_b^2)+0.5',
      inference='eval/no_grad, float32, TF32 matmul/cudnn enabled matching upstream SAM helper')
def raft():
    from scene.utils import get_flow_model,get_flow
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    model=get_flow_model(str(CKPT),str(CFG));assert not model.training
    manifest_path=ROOT/'manifests/teacher_manifest.json'
    record=dict(status='IN_PROGRESS',protocol='dev90',models=dict(raft=str(CKPT),raft_sha256=sha(CKPT),
       raft_config=str(CFG),config_sha256=sha(CFG),sam2=str(ROOT/'weights/sam2.1_hiera_large.pt'),
       sam2_sha256=sha(ROOT/'weights/sam2.1_hiera_large.pt')),flow_records=[],sam_records=[],
       b0_preserved='Lossless FP32 flow/heatmap, original call path and threshold; SAM uses cross-camera ordering.',
       shared_protocol='Separate FB mask and axis resize correction. No B0 mask substitution.',
       formal300_cache='NOT_GENERATED',teacher_gpu_unloaded=False)
    if manifest_path.exists():record=json.loads(manifest_path.read_text(encoding='utf-8'))
    save(manifest_path,record);started=time.perf_counter()
    for scene in DEV:
        for cam in FLOW[scene]:
            ready=ROOT/f'manifests/extraction/{scene}/{cam}.json'
            while not ready.exists():time.sleep(2)
            d=CACHE/f'{scene}/{cam}';d.mkdir(parents=True,exist_ok=True)
            pairs=[(i,i+1) for i in range(89)]+[(0,3)]
            for a,b in pairs:
                out=pair_path(scene,cam,a,b);rev=pair_path(scene,cam,b,a)
                if all(p.exists() and p.with_suffix('.json').exists() for p in [out,rev]):continue
                with torch.no_grad():
                    ia=image(scene,cam,a);ib=image(scene,cam,b)
                    f,fi,fh=get_flow(str(CKPT),str(CFG),model,ia,ib)
                    r,ri,rh=get_flow(str(CKPT),str(CFG),model,ib,ia)
                    assert torch.isfinite(f).all() and torch.isfinite(r).all()
                    sf=f.clone();sr=r.clone();sf[:,1]*=507/506;sr[:,1]*=507/506
                    checks=[]
                    for p,x,info,heat,sx,other,source,target in [(out,f,fi,fh,sf,sr,a,b),(rev,r,ri,rh,sr,sf,b,a)]:
                        mask,err=fb_mask(sx,other)
                        fp16=sx.half().float();error=(fp16-sx).abs()
                        m=metadata(scene,cam,source,target)
                        m.update(fp16_abs_max=error.max().item(),fp16_mean_epe=(fp16-sx).norm(dim=1).mean().item(),
                                 shared_coverage=mask.float().mean().item(),motion_coverage=(x[0].norm(dim=0)>1.5).float().mean().item())
                        np.savez_compressed(p,b0_flow=x[0].cpu().numpy(),b0_heatmap=heat[0].cpu().numpy(),
                          shared_flow=sx[0].half().cpu().numpy(),shared_valid=mask.cpu().numpy(),
                          b0_motion=(x[0].norm(dim=0)>1.5).cpu().numpy(),info=info[0].half().cpu().numpy(),
                          metadata=np.array(json.dumps(m)))
                        checks.append(m)
                    if a in [0,44,88] and b==a+1:
                        repeat,_,_=get_flow(str(CKPT),str(CFG),model,ia,ib)
                        replay=torch.from_numpy(np.load(out)['b0_flow']).cuda()[None]
                        checks[0]['online_repeat_abs_max']=(repeat-replay).abs().max().item()
                        checks[0]['cache_roundtrip_abs_max']=(f-replay).abs().max().item()
                    for p,m in zip([out,rev],checks):
                        m['path']=str(p);m['sha256']=sha(p)
                        save(p.with_suffix('.json'),m)
                del ia,ib,f,fi,fh,r,ri,rh,sf,sr
            print('RAFT PASS',scene,cam,flush=True)
    del model;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    record['flow_records']=[str(p) for p in sorted(CACHE.glob('*/*/*.json')) if not p.name.startswith('sam_')]
    record.update(raft_status='PASS',raft_seconds=time.perf_counter()-started,raft_peak_allocated=torch.cuda.max_memory_allocated(),
                  raft_gpu_unloaded=True);save(manifest_path,record)
def sam():
    from scene.utils import get_sam2_model,get_sam2_mask
    manifest_path=ROOT/'manifests/teacher_manifest.json';record=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert record['raft_status']=='PASS'
    predictor=get_sam2_model(str(ROOT/'weights/sam2.1_hiera_large.pt'),'configs/sam2.1/sam2.1_hiera_l.yaml')
    assert not predictor.training
    mask=motion=final=repeat=None
    started=time.perf_counter();refresh=[1,2]+list(range(10,91,10))
    for scene in DEV:
        for step in refresh:
            current=step-1;a,b=(0,3) if step==1 else (current-1,current)
            outputs=[CACHE/f'{scene}/{cam}/sam_{step:04d}.npz' for cam in FLOW[scene]]
            if all(p.exists() and p.with_suffix('.json').exists() for p in outputs) and (step!=1 or scene in record.get('sam_online_checks',{})):
                continue
            images=[image(scene,cam,current) for cam in FLOW[scene]]
            masks=[torch.from_numpy(np.load(pair_path(scene,cam,a,b))['b0_motion']).cuda() for cam in FLOW[scene]]
            with torch.no_grad():result=get_sam2_mask(predictor,images,masks)
            assert len(result)==4
            for cam,p,mask,motion in zip(FLOW[scene],outputs,result,masks):
                final=mask+motion
                assert final.dtype==torch.bool  # matches upstream boolean + boolean union
                m=dict(scene=scene,camera=cam,training_step=step,rgb_frame=current,flow_pair=[a,b],
                   camera_order=FLOW[scene],threshold_pixels=1.5,precision='binary lossless',
                   refresh_interval_steps=10,held_until_next_refresh=True,sam_coverage=mask.float().mean().item(),
                   union_coverage=final.float().mean().item(),b0_behavior='Upstream init_state3 cross-view list; all four views prompted',
                   checkpoint_sha256=record['models']['sam2_sha256'],
                   config_sha256=sha(WORK/'sam2/configs/sam2.1/sam2.1_hiera_l.yaml'),
                   code_sha256=sha(WORK/'sam2/sam2_video_predictor.py'),
                   shape_hw=[507,676],mask_frame_grid='current RGB frame; upstream previous-source flow prompt preserved')
                np.savez_compressed(p,b0_sam=mask.cpu().numpy(),b0_union=final.cpu().numpy(),metadata=np.array(json.dumps(m)))
                m.update(path=str(p),sha256=sha(p));save(p.with_suffix('.json'),m)
            if step==1:
                with torch.no_grad():repeat=get_sam2_mask(predictor,images,masks)
                record.setdefault('sam_online_checks',{})[scene]=[int((x!=y).sum().item()) for x,y in zip(result,repeat)]
                save(manifest_path,record)
            del images,masks,result
            print('SAM PASS',scene,step,flush=True)
    del predictor,mask,motion,final,repeat
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    record['sam_records']=[str(p) for p in sorted(CACHE.glob('*/*/sam_*.json'))]
    record.update(sam_status='PASS',sam_seconds=time.perf_counter()-started,sam_peak_allocated=torch.cuda.max_memory_allocated(),
                  status='PASS',teacher_gpu_unloaded=True,remaining_process_allocated=torch.cuda.memory_allocated())
    save(manifest_path,record)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['raft','sam']);a=p.parse_args()
    torch.manual_seed(0);torch.cuda.reset_peak_memory_stats();globals()[a.mode]()
