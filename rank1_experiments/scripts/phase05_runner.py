"""Autonomous serial seed0 sweep; fail closed, checkpoint resume and Git trial history."""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'working/MoRGS')]
from phase05.common import read,write,sha,now,state_update,protocol
from phase05_launch import launch

def lines(path):
    if not path.exists():return []
    values=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:values.append(json.loads(line))
        except json.JSONDecodeError:break # last line may be in-flight
    return values

class Runner:
    def __init__(self):
        self.p=protocol();self.active={};self.done=[];self.stopped=threading.Event()
        self.status='RUNNING';self.started=now()

    def progress(self):
        while not self.stopped.wait(20):self.save_progress()

    def save_progress(self):
        job=self.active.copy();frames=lines(Path(job['run'])/'resources.jsonl') if job.get('run') else []
        write(ROOT/'manifests/phase05_progress.json',dict(status=self.status,pid=os.getpid(),started=self.started,updated=now(),
              active=job,active_complete_frame=frames[-1]['frame'] if frames else None,completed=self.done.copy(),
              planned_search_jobs=24,completed_search_jobs=sum(x.get('method')!='F0' for x in self.done),
              scope='seed0 search only; final three-seed gate not passed',next_phase='05'))

    def invoke(self,label,script,*args):
        if launch(label,ROOT/'scripts'/script,*args)!=0:raise RuntimeError('Actual command failed: '+label)

    def verify_frozen(self):
        assert sha(ROOT/'manifests/phase05_protocol.json')==read(ROOT/'experiment_state.json')['phase05_protocol_sha256']
        assert all(sha(ROOT/p)==h for p,h in self.p['immutable_sources'].items())
        assert all(sha(ROOT/p)==h for p,h in read(ROOT/'manifests/phase05_code_lock.json').items())
        assert all(sha(ROOT/p)==h for p,h in read(ROOT/'manifests/phase05_calibration_lock.json').items())

    def job(self,job,calibrate=False):
        self.verify_frozen();self.active=dict(scene=job['scene'],method=job['method'],ratio=job['ratio'],run=job['run'],stage='training')
        self.save_progress();run=Path(job['run']);scene=job['scene'];method=job['method'];trial=job['trial']
        assert sha(job['config'])==job['config_sha256']
        assert sha(job['initial']['path'])==job['initial']['sha256']
        index=ROOT/'manifests/phase05_trials'/f'{scene}_{method}_{trial}.json'
        if index.exists():
            row=read(index);assert read(run/'evaluation/validation/results.json')['frames']==90
            self.done.append({k:row[k] for k in ('scene','method','ratio','run')});return
        if calibrate and (run/'calibration.json').exists():
            assert read(run/'calibration.json')['status']=='PASS';return
        complete=2 if calibrate else 90
        resource=lines(run/'resources.jsonl');initial=Path(job['initial']['path'])
        start=initial
        if resource:
            import torch
            last=torch.load(run/'checkpoints/latest.pt',weights_only=False,map_location='cpu')
            assert last['frame']==resource[-1]['frame']
            assert lines(run/'training_metrics.jsonl')[-1]['frame']==last['frame'],'Incomplete boundary diagnostics; explicit recovery required'
            if last['frame']<complete:start=run/'checkpoints/latest.pt'
            del last
        if not resource or resource[-1]['frame']<complete:
            assert shutil.disk_usage(ROOT).free>80*1024**3,'Disk safety margin reached; preserve all artifacts'
            cal=next(j for j in self.p['jobs'] if j['scene']==scene and j['role']=='calibration')
            calpath=Path(cal['run'])/'calibration.json'
            value=0. if method in ('F0','CALIBRATE') else job['ratio']*read(calpath)['methods'][method]['unit_lambda']
            args=['--config',job['config'],'--start_checkpoint',str(start),'--stop_after_frame',str(complete),
                  '--flow_method','F0' if calibrate else method,'--flow_k','64','--flow_sigma','2',
                  '--flow_lambda',str(value),'--flow_grad_ratio',str(job['ratio']),'--pilot_calibration_file',str(calpath)]
            if calibrate:args.append('--pilot_calibrate')
            self.invoke(f'train_{scene}_{method}_{trial}','../working/MoRGS/train_phase05.py',*args)
        if calibrate:
            assert read(run/'calibration.json')['status']=='PASS';return
        resources=lines(run/'resources.jsonl');assert [r['frame'] for r in resources]==list(range(2,91))
        train=lines(run/'training_metrics.jsonl');assert [r['frame'] for r in train]==list(range(2,91))
        samplers={r['frame']:r['sampler_sha256'] for r in train}
        samplerpath=ROOT/'manifests'/f'phase05_sampler_{scene}.json'
        if samplerpath.exists():assert {str(k):v for k,v in samplers.items()}==read(samplerpath)
        else:write(samplerpath,samplers)
        cfg=read(run/'resolved_config.json');flow=lines(run/'flow_diagnostics.jsonl')
        assert not cfg['flow_freeze_topology'] and cfg['max_frames']==90 and cfg['seed']==0
        assert cfg['use_xyz_legacy'] is False and cfg['test_indices']==[]
        assert all(r['iterations']==(128 if scene=='coffee_martini' else 144) for r in resources)
        if method!='F0':
            assert all(r['lambda_flow']==cfg['effective_flow_lambda'] and math.isfinite(r['flow_loss']) for r in flow)
            assert min(r['uid_mass_retention'] for r in flow)>=.95,'UID coverage failure; stop scaling'
            assert min(r['topk_mass_retention'] for r in flow)>=.95,'K truncation failure; stop and repair all controls'
            assert min(r['candidate_coverage'] for r in flow)>=.9,'Candidate coverage failure'
        else:assert all(r['flow_loss']==0 and r['flow_grad_norm']==0 for r in flow)
        import torch
        checkpoint=torch.load(run/'checkpoints/latest.pt',weights_only=False,map_location='cpu')
        state=checkpoint['gaussian_state'];identity=state['identity']
        assert checkpoint['frame']==90 and state['phase04_flow_lambda']==cfg['effective_flow_lambda']
        assert identity.uid.unique().numel()==len(identity.uid)==resources[-1]['points']
        assert state['topology_version']>0,'Native density must actually execute'
        for group in state['optimizer'].param_groups:
            if group['name'] in state['param_names']:assert group['params'][0] is state['_latents'][group['name']]
            if group['name']=='offset_3d':assert group['params'][0] is state['offset_3d']
        del checkpoint,state,identity
        self.active['stage']='validation';self.save_progress()
        result=run/'evaluation/validation/results.json'
        csvpath=run/'metrics_per_frame.csv'
        if not result.exists() or 'PSNR' not in csvpath.read_text().splitlines()[0]:
            self.invoke(f'eval_{scene}_{method}_{trial}','phase05_evaluate.py','--run',run,'--scene',scene)
        evaluation=read(result);assert evaluation['frames']==90 and evaluation['status']=='PASS'
        with csvpath.open(encoding='utf-8') as f:assert len(list(csv.DictReader(f)))==90
        ckpts=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((run/'checkpoints').glob('*.pt'))]
        row=dict(status='PASS',scene=scene,method=method,ratio=job['ratio'],sigma=2,k=64,seed=0,run=str(run.relative_to(ROOT)),
                 metrics=evaluation['unified_fp32'],regions=evaluation['regions'],updated_frames=89,
                 seconds=sum(r['train_and_save_seconds'] for r in resources),
                 peak_allocated_gib=max(r['peak_allocated'] for r in resources)/1024**3,
                 peak_reserved_gib=max(r['peak_reserved'] for r in resources)/1024**3,
                 disk_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),
                 lambda_flow=cfg['effective_flow_lambda'],initial=job['initial'],config_sha256=job['config_sha256'],
                 sampler_manifest_sha256=sha(samplerpath),checkpoints=ckpts,
                 uid_min=None if method=='F0' else min(r['uid_mass_retention'] for r in flow),
                 k_min=None if method=='F0' else min(r['topk_mass_retention'] for r in flow),
                 candidate_coverage_min=None if method=='F0' else min(r['candidate_coverage'] for r in flow),
                 evaluation_sha256=sha(result),completed=now())
        write(index,row);self.done.append({k:row[k] for k in ('scene','method','ratio','run')});self.save_progress()
        subprocess.run([sys.executable,str(ROOT/'scripts/phase05_report.py')],check=True)
        self.commit_trial(index,scene,method,trial)

    def commit_trial(self,index,scene,method,trial):
        # Explicit small paths only; no accidental dataset/checkpoint staging.
        paths=[index,ROOT/'reports/phase05_go_no_go.md',ROOT/'manifests/phase05_progress.json',
               ROOT/'manifests/phase05_commands.jsonl',ROOT/'manifests/phase05_started_commands.jsonl',
               ROOT/'manifests'/f'phase05_sampler_{scene}.json']
        subprocess.run(['git','add',*[str(p) for p in paths if p.exists()]],cwd=ROOT.parent,check=True)
        subprocess.run(['git','commit','-m',f'phase05 trial: {scene} {method} {trial} full90 validation and checkpoint index'],cwd=ROOT.parent,check=True)

    def run(self):
        self.active=dict(stage='freeze independent validation proxies');self.save_progress()
        self.invoke('regions','phase05_regions.py')
        for job in self.p['jobs']:
            if job['role']=='calibration':self.job(job,True)
        for scene in self.p['scenes']:
            output=ROOT/'runs/pilot/references/B0'/scene/'evaluation/validation'
            if not (output/'results.json').exists():
                self.active=dict(stage='baseline reference evaluation',scene=scene);self.save_progress()
                self.invoke('reference_'+scene,'phase05_evaluate.py','--run',ROOT/'runs/B0/dev'/scene/'seed0_loader_exact','--scene',scene,'--output',output)
        for job in self.p['jobs']:
            if job['method']=='F0':self.job(job)
        for ratio in self.p['ratios_in_order']:
            for method in ('M','B2','H','B1'):
                for scene in self.p['scenes']:
                    self.job(next(j for j in self.p['jobs'] if j['scene']==scene and j['method']==method and j['ratio']==ratio))
        subprocess.run([sys.executable,str(ROOT/'scripts/phase05_report.py')],check=True)
        gate=read(ROOT/'manifests/phase05_screen.json')
        if gate['signal']:
            self.status='SEED0_SIGNAL_REPEAT_REQUIRED'
            state_update(status='RUNNING',phase05='REPEAT_REQUIRED',continue_gate_passed=False,next_phase='05',
                         next_actions=['Prepare formal cam01 data and shared-head resource probe','Run selected M and two controls over both300-frame scenes, seeds0/1/2; fixed global lambdas'],
                         scope='Seed0 screening signal only; mandatory300-frame3-seed repetition remains')
        else:
            self.invoke('one_readonly_diagnosis','phase05_diagnose.py')
            self.status='NO_GO_AFTER_READONLY_DIAGNOSIS'
            state_update(status='NO_GO',phase05='COMPLETE_NO_GO',continue_gate_passed=False,next_phase='STOPPED',
                         next_actions=['Review paired negative results and one read-only diagnosis; no stage06/08 expansion'],
                         scope='All24 seed0 trials and one diagnosis complete; no screening signal, no expansion')
        self.active={};self.save_progress()
        self.invoke('paired_figures','phase05_figures.py')
        subprocess.run([sys.executable,str(ROOT/'scripts/phase05_report.py')],check=True)
        paths=list((ROOT/'manifests').glob('phase05*.json'))+list((ROOT/'manifests').glob('phase05*.jsonl'))
        paths+=[ROOT/'manifests/phase05_trials',ROOT/'reports/phase05_go_no_go.md',ROOT/'experiment_state.json',ROOT/'configs/frozen_pilot_configs',ROOT/'diagnostics/phase05']
        subprocess.run(['git','add',*[str(p) for p in paths if p.exists()]],cwd=ROOT.parent,check=True)
        subprocess.run(['git','commit','-m','phase05 seed0 screening complete: preserve all trials and explicit repeat/no-go decision'],cwd=ROOT.parent,check=True)

def main():
    lock=ROOT/'manifests/phase05_runner.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Runner lock exists; verify recorded PID before explicit resume')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    runner=Runner();thread=threading.Thread(target=runner.progress,daemon=True);thread.start()
    runner.save_progress()
    subprocess.run([sys.executable,str(ROOT/'scripts/phase05_report.py')],check=True)
    try:runner.run()
    except BaseException as error:
        runner.status='FAILED';runner.save_progress()
        write(ROOT/'manifests/phase05_failure.json',dict(at=now(),error=repr(error),active=runner.active,scope='Failed or coverage gate failed; preserve partial run and logs'))
        state_update(status='FAILED',phase05='FAILED',continue_gate_passed=False,next_phase='05',phase05_blocking_reason=repr(error))
        subprocess.run([sys.executable,str(ROOT/'scripts/phase05_report.py')])
        raise
    finally:runner.stopped.set();thread.join();lock.unlink(missing_ok=True)

if __name__=='__main__':main()
