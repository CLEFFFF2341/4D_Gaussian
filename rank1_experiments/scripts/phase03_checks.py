"""CUDA resume variability and snapshot render equivalence checks."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'working/MoRGS'))
import torch
from scene import GaussianModel
from scene.dataset_readers import readCamerasFromPoseBounds
from utils.camera_utils import loadSequentialCam
from gaussian_renderer import render_mask
from phase03_evaluate import RenderSnapshot


def rendering_check(run,frame):
    checkpoint=torch.load(run/'checkpoints'/f'frame_{frame:04d}.pt',weights_only=False)
    state=checkpoint['gaussian_state']
    gaussian=GaussianModel(state['max_sh_degree'],state['latent_args'],state['model_args'])
    gaussian.__dict__.update(state)
    gaussian.setup_functions()
    offset_identity = None
    for group in gaussian.optimizer.param_groups:
        if group['name'] in gaussian.param_names:
            assert group['params'][0] is gaussian._latents[group['name']],group['name']
        if group['name']=='offset_3d':offset_identity = group['params'][0] is gaussian.offset_3d
    source=Path(checkpoint['source_path'])
    meta=json.loads((source/'dataset_meta.json').read_text(encoding='utf-8'))
    infos,_=readCamerasFromPoseBounds(str(source/'poses_bounds.npy'),meta['size_wh'],N_video_views=10)
    camera=loadSequentialCam(SimpleNamespace(data_device='cuda',znear=5.,zfar=300.),0,infos[0],1.)
    pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
    bg=torch.zeros(3,device='cuda')
    snapshot=torch.load(run/'snapshots'/f'{frame:04d}.pt',weights_only=False)
    decoded=RenderSnapshot(snapshot)
    with torch.no_grad():
        live=render_mask(camera,gaussian,pipe,bg)['render']
        restored=render_mask(camera,decoded,pipe,bg)['render']
    error=float((live-restored).abs().max())
    assert error<1e-6,error
    return dict(frame=frame,render_abs_max=error,latent_optimizer_parameter_identity='PASS',
                offset_optimizer_parameter_identity=offset_identity,
                offset_note='Author prune_points slices offset instead of assigning optimizer replacement; preserved in B0 reference.')


def differences(first,second):
    result={}
    for name,value in first['attributes'].items():
        other=second['attributes'][name]
        result[name]=float((value-other).abs().max()) if value.shape==other.shape else None
    return result

def compare_renders(first,second,third,source):
    meta=json.loads((source/'dataset_meta.json').read_text(encoding='utf-8'))
    infos,_=readCamerasFromPoseBounds(str(source/'poses_bounds.npy'),meta['size_wh'],N_video_views=10)
    models=[RenderSnapshot(x) for x in [first,second,third]]
    pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
    results=[]
    with torch.no_grad():
        for index in [0,3,7,12]:
            camera=loadSequentialCam(SimpleNamespace(data_device='cuda',znear=5.,zfar=300.),index,infos[index],1.)
            images=[render_mask(camera,m,pipe,torch.zeros(3,device='cuda'))['render'].clamp(0,1) for m in models]
            error=float((images[0]-images[1]).abs().mean())
            noise=float((images[1]-images[2]).abs().mean())
            tolerance=max(1e-5,2*noise)
            # Absolute cap: one RGB intensity level on average; measured CUDA
            # rerun variation supplies the tighter relative resume gate.
            assert error<=tolerance and error<1/255,(index,error,noise,tolerance)
            results.append(dict(camera=meta['cameras'][index],resume_rgb_mae=error,
                repeat_rgb_mae=noise,tolerance=tolerance,
                resume_rgb_abs_max=float((images[0]-images[1]).abs().max())))
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--continuous',type=Path,required=True)
    parser.add_argument('--resumed',type=Path)
    parser.add_argument('--repeat',type=Path)
    args=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    checks=[rendering_check(args.continuous,f) for f in [1,2,5,10]]
    report=dict(snapshot_checks=checks,continuous_run=str(args.continuous),
                resumed_run=str(args.resumed),repeat_run=str(args.repeat))
    if args.resumed:
        first=torch.load(args.continuous/'snapshots/0010.pt',weights_only=False)
        second=torch.load(args.resumed/'snapshots/0010.pt',weights_only=False)
        third=torch.load(args.repeat/'snapshots/0010.pt',weights_only=False)
        error=differences(first,second)
        variation=differences(second,third)
        counts=[x['attributes']['xyz'].shape[0] for x in [first,second,third]]
        count_tolerance=max(16,2*abs(counts[1]-counts[2]))
        assert abs(counts[0]-counts[1])<=count_tolerance,(counts,count_tolerance)
        checkpoint=torch.load(args.continuous/'checkpoints/frame_0005.pt',weights_only=False)
        renders=compare_renders(first,second,third,Path(checkpoint['source_path']))
        report.update(resume_attribute_abs_max=error,cuda_repeat_attribute_abs_max=variation,
                      point_counts=counts,point_count_tolerance=count_tolerance,render_comparisons=renders,
                      topology_note='Repeat-run variability affects threshold densification; compare rendered fields when topology differs. Root cause not isolated.')
        for run in [args.resumed,args.repeat]:
            ck=torch.load(run/'checkpoints/frame_0010.pt',weights_only=False)
            g=ck['gaussian_state']
            assert g['offset_3d'] is not None and g['mapping'] is not None
            assert g['prev_atts']['xyz'] is not None and g['prev_latents']['f_dc'] is not None
        report.update(resume_check='PASS',status='PASS')
    else:report['status']='SNAPSHOT_PASS_ONLY'
    (ROOT/'manifests/phase03_resume_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
