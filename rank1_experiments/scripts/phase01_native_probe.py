"""Bounded native rasterizer diagnostics, not a baseline training reproduction."""
import json
import argparse
import sys
import time
from pathlib import Path
import torch
import gaussian_rasterization_grad as native

ROOT=Path(__file__).resolve().parents[1]

def render(mask):
    device='cuda'
    means=torch.tensor([[-.25,0.,3.],[0.,.1,3.],[.25,0.,3.]],device=device,requires_grad=True)
    flow=torch.tensor([[.1,0.,0.]]*3,device=device,requires_grad=True)
    screen=torch.zeros_like(means,requires_grad=True)
    shs=torch.tensor([[[1.,.2,.1]]]*3,device=device,requires_grad=True)
    opacity=torch.full((3,1),.8,device=device,requires_grad=True)
    scales=torch.full((3,3),.1,device=device,requires_grad=True)
    rotations=torch.tensor([[1.,0.,0.,0.]]*3,device=device,requires_grad=True)
    motion=torch.full((3,1),.5,device=device,requires_grad=True)
    P=torch.zeros((4,4),device=device)
    P[0,0]=P[1,1]=1.
    P[3,2]=1.
    P[2,2]=100./99.9
    P[2,3]=-10./99.9
    settings=native.GaussianRasterizationSettings(image_height=48,image_width=64,tanfovx=1.,tanfovy=1.,
        bg=torch.zeros(3,device=device),scale_modifier=1.,viewmatrix=torch.eye(4,device=device),
        projmatrix=P.T.contiguous(),sh_degree=0,campos=torch.zeros(3,device=device),
        prefiltered=False,debug=True,render_depth=True,backward_alpha=True,render_flow=True,render_motion_label=True)
    outputs=native.GaussianRasterizer(settings)(means3D=means,flow3D=flow,means2D=screen,shs=shs,
        motion_label=motion,opacities=opacity,scales=scales,rotations=rotations,update_mask=mask)
    color, rendered_motion, image_flow, influence, count, depth, alpha, radii=outputs
    assert torch.isfinite(color).all() and torch.isfinite(image_flow).all()
    normalized=image_flow/(alpha+1e-6)
    visible=alpha.squeeze(0)>.1
    flow_x=normalized[0][visible].mean().item()
    loss=color.square().mean()+image_flow.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    return dict(flow_x_mean=flow_x,radii=radii.tolist(),visible_pixels=visible.sum().item(),
                loss=loss.item(),means_gradient=means.grad.tolist(),flow_gradient=flow.grad.tolist(),
                finite_gradients=bool(torch.isfinite(means.grad).all() and torch.isfinite(flow.grad).all()))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['pristine','mask_stride_8'],default='pristine')
    parser.add_argument('--output-name', default=None)
    args=parser.parse_args()
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    all_true=render(torch.ones((3,8),device='cuda',dtype=torch.bool))
    selective=torch.zeros((3,8),device='cuda',dtype=torch.bool)
    selective[0,7]=True  # Only point 0's motion-label column; every xyz/flow column is false.
    masked=render(selective)
    expected=64./2*.1/3.
    result=dict(mode=f'Python {sys.version.split()[0]} / torch {torch.__version__} / CUDA {torch.version.cuda}', source_variant=args.variant,
        module_path=native.__file__,torch=torch.__version__,torch_cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        all_true=all_true,selective_mask=selective.tolist(),selective_result=masked,
        exact_center_projection_delta_x=expected,native_to_exact_ratio=all_true['flow_x_mean']/expected,
        seconds=time.perf_counter()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        note='Tiny diagnostic only. Full upstream teacher/training/checkpoint/evaluation path remains untested.')
    if args.variant=='mask_stride_8':
        reference=json.loads((ROOT/'manifests/native_probe_pristine.json').read_text(encoding='utf-8'))
        assert abs(reference['all_true']['loss']-all_true['loss'])<1e-7
        assert torch.tensor(masked['means_gradient']).abs().max().item()==0
        result['mask_stride_regression_passed']=True
    output_name=args.output_name or ('native_probe_'+args.variant+'.json')
    assert Path(output_name).name == output_name
    (ROOT/'manifests'/output_name).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
