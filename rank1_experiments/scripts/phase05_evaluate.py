"""Fixed cam01 full/region metrics, isolated from optimization and old manifests."""
import argparse
import csv
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'working/MoRGS')]
from phase05.common import read,write,sha,protocol
from phase04_evaluate import RenderSnapshot,psnr
from gaussian_renderer import render_mask
from scene.dataset_readers import readCamerasFromPoseBounds
from utils.camera_utils import loadSequentialCam
from utils.loss_utils import ssim,create_window
from lpipsPyTorch.modules.lpips import LPIPS
from phase04.model import SharedGaussianModel

def ssim_map(a,b):
    window=create_window(11,3).to(a)
    conv=lambda x:F.conv2d(x,window,padding=5,groups=3)
    ma,mb=conv(a),conv(b)
    va,vb=conv(a*a)-ma*ma,conv(b*b)-mb*mb
    cov=conv(a*b)-ma*mb
    return (((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean(1)[0]

def region_metrics(error,ssmap,mask):
    count=int(mask.sum())
    if not count:return dict(pixels=0,PSNR=None,SSIM=None)
    mse=float(error[mask].mean())
    return dict(pixels=count,PSNR=-10*math.log10(max(mse,1e-30)),SSIM=float(ssmap[mask].mean()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--scene',required=True)
    p.add_argument('--output',type=Path);p.add_argument('--frames',default='');p.add_argument('--smoke-regions',action='store_true')
    a=p.parse_args();proto=protocol()
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    criterion=LPIPS(net_type='vgg',version='0.1').cuda().eval()
    data=ROOT/'data/dynerf/dev/validation'/a.scene;meta=read(data/'dataset_meta.json');assert meta['cameras']==['cam01']
    infos,_=readCamerasFromPoseBounds(str(data/'poses_bounds.npy'),meta['size_wh'],N_video_views=10)
    camera=loadSequentialCam(SimpleNamespace(data_device='cuda',znear=5.,zfar=300.),0,infos[0],1.)
    pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False);background=torch.zeros(3,device='cuda')
    frames=[int(f) for f in a.frames.split(',')] if a.frames else list(range(1,91))
    out=a.output or a.run/'evaluation/validation';out.mkdir(parents=True,exist_ok=True)
    records=[];checks=[];render_times=[]
    maskroot=ROOT/('diagnostics/phase05/smoke_regions' if a.smoke_regions else 'cache/phase05_eval_regions/dev')/a.scene
    if not a.smoke_regions:assert read(ROOT/'manifests/phase05_regions.json')['status']=='FROZEN'
    for frame in frames:
        path=a.run/'snapshots'/f'{frame:04d}.pt'
        if frame==1 and not path.exists():
            path=ROOT/'runs/B0/dev'/a.scene/'seed0_loader_exact/snapshots/0001.pt'
        snapshot=torch.load(path,weights_only=False);model=RenderSnapshot(snapshot)
        with Image.open(data/'cam01'/f'{frame-1:04d}.png') as image:gt=to_tensor(image.convert('RGB')).cuda()[None]
        with np.load(maskroot/f'{frame-1:04d}.npz') as masks:regions={k:torch.from_numpy(masks[k].copy()).cuda() for k in masks.files}
        with torch.no_grad():
            torch.cuda.synchronize();tic=time.perf_counter()
            prediction=render_mask(camera,model,pipe,background)['render'].clamp(0,1)[None]
            torch.cuda.synchronize();render_times.append(time.perf_counter()-tic)
            values=dict(PSNR=float(psnr(prediction,gt)),SSIM=float(ssim(prediction,gt)),LPIPS=float(criterion(prediction*2-1,gt*2-1)))
            error=(prediction-gt).square().mean(1)[0];ssmap=ssim_map(prediction,gt)
            assert abs(float(ssmap.mean())-values['SSIM'])<1e-6
            regional={k:region_metrics(error,ssmap,mask) for k,mask in regions.items()}
            cp=a.run/'checkpoints'/f'frame_{frame:04d}.pt'
            if frame in (2,30,60,90) and cp.exists():
                stored=torch.load(cp,weights_only=False);state=stored['gaussian_state']
                full=SharedGaussianModel(state['max_sh_degree'],state['latent_args'],state['model_args'])
                full.__dict__.update(state);full.setup_functions()
                reference=render_mask(camera,full,pipe,background)['render'].clamp(0,1)[None]
                diff=float((reference-prediction).abs().max());assert diff<1e-6
                checks.append(dict(frame=frame,abs_max=diff));del full,stored,state,reference
            if frame in (1,15,30,45,60,75,90):
                folder=out/'renders';folder.mkdir(exist_ok=True)
                Image.fromarray((prediction[0].permute(1,2,0)*255).round().byte().cpu().numpy()).save(folder/f'{frame-1:04d}.png')
        records.append(dict(frame=frame,source_frame=frame-1,camera='cam01',metrics=values,regions=regional))
        del snapshot,model,prediction,gt,regions,error,ssmap
    assert all(math.isfinite(v) for r in records for v in r['metrics'].values())
    regional={}
    for name in records[0]['regions']:
        subset=[r['regions'][name] for r in records if r['regions'][name]['pixels']]
        regional[name]=dict(nonempty_frames=len(subset),empty_frames=len(records)-len(subset),
                           pixels=sum(r['pixels'] for r in subset),
                           **{k:sum(r[k] for r in subset)/len(subset) if subset else None for k in ('PSNR','SSIM')})
    summary=dict(status='PASS',frames=len(records),role='validation',camera='cam01',
                 unified_fp32={k:sum(r['metrics'][k] for r in records)/len(records) for k in ('PSNR','SSIM','LPIPS')},
                 regions=regional,full_model_snapshot_checks=checks,
                 protocol_sha256=sha(ROOT/'manifests/phase05_protocol.json'),
                 seconds=time.perf_counter()-start,rendering_fps=len(render_times)/sum(render_times),
                 peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    write(out/'per_frame.json',records);write(out/'results.json',summary)
    csvrows=[]
    training={r['frame']:r for r in (read_lines(a.run/'training_metrics.jsonl'))}
    resources={r['frame']:r for r in (read_lines(a.run/'resources.jsonl'))}
    for r in records:
        row=dict(frame=r['frame'],camera='cam01',**r['metrics'],**training.get(r['frame'],{}),**resources.get(r['frame'],{}))
        for name,values in r['regions'].items():row.update({name+'_'+k:v for k,v in values.items()})
        csvrows.append(row)
    fields=list(dict.fromkeys(k for row in csvrows for k in row))
    csvpath=out/'metrics_per_frame.csv' if a.output else a.run/'metrics_per_frame.csv'
    with csvpath.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(csvrows)
    print(__import__('json').dumps(summary))

def read_lines(path):
    return [__import__('json').loads(line) for line in path.read_text().splitlines()] if path.exists() else []

if __name__=='__main__':main()
