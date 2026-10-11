"""Numerical native-kernel and codec smoke tests in the pinned environment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'working/MoRGS'))
os.environ['TORCH_EXTENSIONS_DIR']=str(ROOT/'cache/torch_extensions')
import torch

def knn():
    from simple_knn._C import distCUDA2
    from torch_cluster import knn_graph
    torch.manual_seed(0)
    x=torch.rand((32,3),device='cuda')
    expected=torch.cdist(x,x).square()
    expected.fill_diagonal_(float('inf'))
    expected=expected.topk(3,largest=False,dim=1).values.mean(dim=1)
    actual=distCUDA2(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-6)
    edges=knn_graph(x,k=3,loop=False)
    assert edges.shape==(2,96)
    assert (edges[0]!=edges[1]).all()
    return dict(status='PASS',simple_knn_max_error=(actual-expected).abs().max().item(),
                torch_cluster_edges=list(edges.shape))

def raster():
    from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
    # This vendored standard rasterizer rejects camera-space z <= 4.0.
    x=torch.tensor([[-.2,0,6.],[.2,.1,6.]],device='cuda',requires_grad=True)
    screen=torch.zeros_like(x,requires_grad=True)
    sh=torch.ones((2,1,3),device='cuda',requires_grad=True)
    opacity=torch.full((2,1),.8,device='cuda',requires_grad=True)
    scale=torch.full((2,3),.2,device='cuda',requires_grad=True)
    rot=torch.tensor([[1.,0,0,0]]*2,device='cuda',requires_grad=True)
    P=torch.zeros((4,4),device='cuda')
    P[0,0]=P[1,1]=1
    P[3,2]=1
    P[2,2]=100/99.9
    P[2,3]=-10/99.9
    settings=GaussianRasterizationSettings(image_height=48,image_width=64,
        tanfovx=1.,tanfovy=1.,bg=torch.zeros(3,device='cuda'),scale_modifier=1.,
        viewmatrix=torch.eye(4,device='cuda'),projmatrix=P.T.contiguous(),
        sh_degree=0,campos=torch.zeros(3,device='cuda'),prefiltered=False,debug=True)
    result=GaussianRasterizer(settings)(means3D=x,means2D=screen,shs=sh,
        opacities=opacity,scales=scale,rotations=rot)
    image=result[0]
    assert torch.isfinite(image).all()
    image.square().mean().backward()
    torch.cuda.synchronize()
    for value in [x,sh,opacity,scale,rot]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
    print(json.dumps(dict(loss=image.square().mean().item(),image_max=image.max().item(),
                          radii=result[1].tolist(),xyz_gradient=x.grad.tolist(),
                          opacity_gradient=opacity.grad.tolist())))
    assert x.grad.abs().sum()>0
    return dict(status='PASS',image_shape=list(image.shape),
                loss=image.square().mean().item(),xyz_gradient_norm=x.grad.norm().item())

def codec():
    import torchac
    cdf=torch.tensor([[0.,.25,.5,.75,1.]]*12,dtype=torch.float32)
    symbols=torch.tensor([0,1,2,3]*3,dtype=torch.int16)
    encoded=torchac.encode_float_cdf(cdf,symbols,check_input_bounds=True)
    decoded=torchac.decode_float_cdf(cdf,encoded)
    assert torch.equal(symbols,decoded)
    return dict(status='PASS',symbols=len(symbols),encoded_bytes=len(encoded),exact_roundtrip=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['knn','raster','codec'])
    args=parser.parse_args()
    start=time.perf_counter()
    result=globals()[args.mode]()
    result.update(torch=torch.__version__,cuda=torch.version.cuda,seconds=time.perf_counter()-start)
    (ROOT/'manifests'/f'official_{args.mode}_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
