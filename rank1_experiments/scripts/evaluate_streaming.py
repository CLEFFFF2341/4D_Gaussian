"""Local adapter: preserve upstream metrics while holding one image pair on CUDA."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'working/MoRGS'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'cache/torch'))
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch.modules.lpips import LPIPS

def evaluate(model_path,result_path):
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    criterion=LPIPS(net_type='vgg',version='0.1').cuda().eval()
    gt_root=model_path/'test/gt'
    render_root=model_path/'test/renders'
    per_frame={}
    per_view={}
    cameras=sorted(p.name for p in gt_root.iterdir() if p.is_dir())
    assert cameras,'No ground-truth camera directories'
    for camera in cameras:
        gt_dir=gt_root/camera
        render_dir=render_root/camera
        names=sorted(p.name for p in render_dir.iterdir() if p.is_file())
        assert names and set(names)=={p.name for p in gt_dir.iterdir() if p.is_file()},camera
        records={key:{} for key in ['PSNR','SSIM','LPIPS']}
        with torch.no_grad():
            for name in names:
                with Image.open(gt_dir/name) as im:
                    gt=to_tensor(im).unsqueeze(0)[:,:3].cuda()
                with Image.open(render_dir/name) as im:
                    render=to_tensor(im).unsqueeze(0)[:,:3].cuda()
                assert gt.shape==render.shape,(camera,name)
                values={'PSNR':psnr(render,gt).item(),'SSIM':ssim(render,gt).item(),
                        'LPIPS':criterion(render,gt).item()}
                for metric,value in values.items():
                    assert metric=='PSNR' or torch.isfinite(torch.tensor(value)),(camera,name,metric)
                    records[metric][name]=value
                del gt,render
        per_frame[camera]=records
        per_view[camera]={metric:torch.tensor(list(values.values())).mean().item()
                          for metric,values in records.items()}
    results={metric:sum(x[metric] for x in per_view.values())/len(per_view)
             for metric in ['PSNR','SSIM','LPIPS']}
    result_path.mkdir(parents=True,exist_ok=True)
    for name,value in [('per_frame.json',per_frame),('per_view.json',per_view),('results.json',results)]:
        (result_path/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    meta=dict(adapter='Local streaming image evaluation; upstream metric definitions unchanged',
              model_path=str(model_path),cameras=len(cameras),
              frames=sum(len(x['PSNR']) for x in per_frame.values()),
              seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),
              peak_reserved=torch.cuda.max_memory_reserved())
    (result_path/'evaluation_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return per_frame,results,meta

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-m','--model-path',type=Path,required=True)
    parser.add_argument('--result-dir',type=Path)
    parser.add_argument('--reference-dir',type=Path)
    args=parser.parse_args()
    frames,results,metadata=evaluate(args.model_path,args.result_dir or args.model_path/'eval_streaming')
    if args.reference_dir:
        reference=json.loads((args.reference_dir/'per_frame.json').read_text(encoding='utf-8'))
        assert frames.keys()==reference.keys()
        errors={key:0. for key in ['PSNR','SSIM','LPIPS']}
        for camera,metrics in frames.items():
            for metric,values in metrics.items():
                assert values.keys()==reference[camera][metric].keys()
                for frame,value in values.items():
                    errors[metric]=max(errors[metric],abs(value-reference[camera][metric][frame]))
        assert errors['PSNR']<1e-5 and errors['SSIM']<1e-6 and errors['LPIPS']<1e-8,errors
        assert abs(results['PSNR']-48.1308036087)<1e-4
        check=dict(status='PASS',max_per_frame_errors=errors,synthetic_fixture=True,
                   baseline_results=False,metadata=metadata)
        (ROOT/'manifests/streaming_evaluation_smoke.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps(dict(results=results,metadata=metadata),indent=2))
