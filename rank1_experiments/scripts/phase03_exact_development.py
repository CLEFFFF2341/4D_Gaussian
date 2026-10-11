"""Repeat B0 with training-loader-exact teacher inputs, retaining earlier diagnostics."""
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/'scripts/run_official.cmd'
TRAIN=ROOT/'working/MoRGS/train_phase03.py'
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')

def job(name,arguments):
    log=ROOT/'logs'/f'phase03_exact_{name}_{stamp}.log'
    command=[str(RUNNER)]+list(map(str,arguments))
    tic=time.perf_counter()
    with log.open('w',encoding='utf-8') as handle:
        completed=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,cwd=ROOT)
    record=dict(name=name,command=command,log=str(log),exit_code=completed.returncode,seconds=time.perf_counter()-tic)
    with (ROOT/'manifests/phase03_commands.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
    assert completed.returncode==0,record

def copy_initial(source,target):
    for sub,name in [('snapshots','0001.pt'),('checkpoints','frame_0001.pt')]:
        destination=target/sub/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        assert not destination.exists(),destination
        shutil.copy2(source/sub/name,destination)
    first=json.loads((source/'resources.jsonl').read_text(encoding='utf-8').splitlines()[0])
    first['reused_from']=str(source)
    (target/'resources.jsonl').write_text(json.dumps(first)+'\n',encoding='utf-8')
    (target/'initial_reuse.json').write_text(json.dumps(dict(source=str(source),
        reason='Initial10000 does not render flow; cached frame1 motion masks are bit-identical or source is new exact smoke.',
        teacher_cache='dev_phase03',standalone_initial_seconds=first['train_and_save_seconds']),indent=2),encoding='utf-8')

def main():
    precision=json.loads((ROOT/'manifests/phase03_cache_input_precision.json').read_text(encoding='utf-8'))
    smoke=ROOT/'runs/B0/smoke/coffee_martini'
    continuous=smoke/'continuous_loader_exact'
    initial_old=smoke/'continuous_v2'
    arguments=[TRAIN,'--config',ROOT/'configs/phase03/smoke_coffee.yaml','-m',continuous]
    if precision['coffee_martini']['initial_motion_mask_changed_pixels']==0:
        copy_initial(initial_old,continuous)
        arguments+=['--start_checkpoint',continuous/'checkpoints/frame_0001.pt']
    job('smoke_continuous',arguments)
    for name in ['resumed_loader_exact','repeat_loader_exact']:
        job(name,[TRAIN,'--config',ROOT/'configs/phase03/smoke_coffee.yaml','-m',smoke/name,
                  '--start_checkpoint',continuous/'checkpoints/frame_0005.pt'])
    job('resume_check',[ROOT/'scripts/phase03_checks.py','--continuous',continuous,
                       '--resumed',smoke/'resumed_loader_exact','--repeat',smoke/'repeat_loader_exact'])
    for scene in ['coffee_martini','cut_roasted_beef']:
        target=ROOT/'runs/B0/dev'/scene/'seed0_loader_exact'
        initial=continuous if scene=='coffee_martini' else ROOT/'runs/B0/dev/cut_roasted_beef/seed0'
        arguments=[TRAIN,'--config',ROOT/'configs/phase03'/f'dev_{scene}.yaml','-m',target]
        if scene=='coffee_martini' or precision[scene]['initial_motion_mask_changed_pixels']==0:
            copy_initial(initial,target)
            arguments+=['--start_checkpoint',target/'checkpoints/frame_0001.pt']
        job('dev_'+scene,arguments)

if __name__=='__main__':main()
