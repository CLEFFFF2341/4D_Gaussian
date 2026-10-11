"""Checks that can affect pilot decisions: region metrics and scene-level gating."""
import sys
import ast
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'working/MoRGS')]
from phase05.common import write
from phase05.selection import select,screen
from phase05_evaluate import region_metrics,ssim_map
from utils.loss_utils import ssim
from phase05_regions import masks,fb

for path in list((ROOT/'phase05').glob('*.py'))+list((ROOT/'scripts').glob('phase05*.py'))+[ROOT/'working/MoRGS/train_phase05.py']:
    ast.parse(path.read_text(encoding='utf-8'),filename=str(path))

generator=torch.Generator().manual_seed(414)
a=torch.rand((1,3,31,37),generator=generator);b=torch.rand(a.shape,generator=generator)
assert abs(float(ssim_map(a,b).mean())-float(ssim(a,b)))<1e-7
assert region_metrics(torch.zeros(31,37),torch.ones(31,37),torch.zeros(31,37,dtype=torch.bool))['PSNR'] is None
metric=region_metrics(torch.full((31,37),.01),torch.ones(31,37),torch.ones(31,37,dtype=torch.bool))
assert abs(metric['PSNR']-20)<1e-5
flow=torch.zeros((1,2,31,37));valid=torch.ones((31,37),dtype=torch.bool)
flow[:,:,8:22,10:28]=1.;regions=masks(flow,valid)
assert regions['dynamic'].sum()>0 and regions['boundary'].sum()>0
assert not (regions['dynamic']&regions['static']).any()
assert (regions['dynamic']|regions['static']).all()
zero=torch.zeros_like(flow);assert fb(zero,zero).all()
f=zero.clone();f[:,0]=2;rev=zero.clone();rev[:,0]=-2
v=fb(f,rev);assert v[:,:-2].all() and not v[:,-2:].any()
rows=[]
for s in ('coffee_martini','cut_roasted_beef'):
    for method in ('B1','B2','H','M'):
        for ratio in (.03,.1,.3):
            rows.append(dict(scene=s,method=method,ratio=ratio,metrics=dict(PSNR=20+(.06 if method=='M' else 0),LPIPS=.2),regions=dict(boundary=dict(PSNR=20))))
w=select(rows);assert all(r['ratio']==.03 for r in w.values())
assert screen(rows,w)['signal'] and not screen(rows,w)['final_repeat_gate_passed']
for row in rows:
    if row['method']=='M':row['metrics']['LPIPS']=.21
assert not screen(rows,select(rows))['signal']
for row in rows:
    if row['method']=='M':row['metrics']['LPIPS']=.2;row['metrics']['PSNR']=19.8 if row['scene']=='cut_roasted_beef' else 20.2
assert not screen(rows,select(rows))['signal'],'One large gain cannot hide the other scene decline'
write(ROOT/'manifests/phase05_tests.json',dict(status='PASS',tests=['SSIM_map_matches_frozen_metric','empty_region_is_missing','known_region_error','fixed_proxy_partition','source_grid_FB_bounds','equal_macro_selection_tie','LPIPS_and_other_scene_guard','seed0_cannot_pass_final_repeat_gate']))
print('PHASE05_TESTS_PASS')
