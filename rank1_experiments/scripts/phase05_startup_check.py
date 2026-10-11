"""Actual common calibration and one-frame training/evaluation smoke before long queue."""
import json
import subprocess
import sys
from pathlib import Path
import yaml
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from phase05.common import protocol,read,write,sha
from phase05_launch import launch

p=protocol();rows=[]
for scene in p['scenes']:
    cal=next(j for j in p['jobs'] if j['scene']==scene and j['role']=='calibration')
    run=Path(cal['run']);out=run/'calibration.json'
    if not out.exists():
        assert launch('calibrate_'+scene,ROOT/'working/MoRGS/train_phase05.py','--config',cal['config'],
                      '--start_checkpoint',cal['initial']['path'],'--stop_after_frame','2','--flow_method','F0',
                      '--flow_lambda','0','--flow_k','64','--pilot_calibrate')==0
    d=read(out);assert d['status']=='PASS'
    prefix=d['sampler_prefix_sha256']
    for method in p['methods']:assert d['methods'][method]['gradient']['xyz_norm']>0 and d['methods'][method]['gradient']['offset_norm']>0
    trial=next(j for j in p['jobs'] if j['scene']==scene and j['method']=='M' and j['ratio']==.1)
    cfg=yaml.safe_load(Path(trial['config']).read_text());smoke=ROOT/'runs/pilot/startup'/scene/'M'
    cfg['model_params']['model_path']=str(smoke)
    path=ROOT/'configs/phase05_startup'/f'{scene}_M.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    value=.1*d['methods']['M']['unit_lambda']
    if not (smoke/'resource.json').exists():
        assert launch('real_smoke_'+scene,ROOT/'working/MoRGS/train_phase05.py','--config',path,
                      '--start_checkpoint',cal['initial']['path'],'--stop_after_frame','2','--flow_method','M',
                      '--flow_lambda',value,'--flow_k','64','--flow_sigma','2','--pilot_calibration_file',out)==0
    resource=read(smoke/'resource.json');assert [r['frame'] for r in resource['frames']]==[2]
    resolved=read(smoke/'resolved_config.json');assert resolved['effective_flow_lambda']==value
    flow=[json.loads(l) for l in (smoke/'flow_diagnostics.jsonl').read_text().splitlines()]
    assert len(flow)>0 and all(r['flow_xyz_grad_norm']>0 and r['flow_offset_grad_norm']>0 for r in flow)
    assert all(r['diagnostic_samples']>0 and len(r['outlier_histogram'])==5 for r in flow)
    cp=torch.load(smoke/'checkpoints/latest.pt',weights_only=False,map_location='cpu');state=cp['gaussian_state']
    for group in state['optimizer'].param_groups:
        if group['name']=='offset_3d':assert group['params'][0] is state['offset_3d']
    assert state['identity'].uid.unique().numel()==len(state['identity'].uid)
    del cp,state
    if not (smoke/'evaluation/validation/results.json').exists():
        assert launch('eval_smoke_'+scene,ROOT/'scripts/phase05_evaluate.py','--run',smoke,'--scene',scene,'--frames','2','--smoke-regions')==0
    evaluation=read(smoke/'evaluation/validation/results.json');assert evaluation['frames']==1 and evaluation['full_model_snapshot_checks'][0]['abs_max']==0
    rows.append(dict(scene=scene,calibration=out.as_posix(),calibration_sha256=sha(out),
                     lambda_flow=value,gradients_nonzero=True,posterior_recorded=True,sampler_prefix_sha256=prefix,
                     peak_allocated_gib=resource['peak_allocated']/1024**3,train_seconds=resource['train_and_save_seconds'],
                     evaluation_status=evaluation['status']))
write(ROOT/'manifests/phase05_startup_check.json',dict(status='PASS',scenes=rows,
     scope='Training and evaluation one-frame startup check; not a full90 search trial and not selection evidence'))
print('PHASE05_STARTUP_CHECK_PASS')
