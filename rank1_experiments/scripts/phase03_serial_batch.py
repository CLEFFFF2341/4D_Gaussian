"""Serialized GPU jobs with durable command/exit records. Stops on any failure."""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/'scripts/run_official.cmd'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['pipeline','after-dev','after-dense','eval-dev','formal-teachers','dense','formal-train','eval-final'],required=True)
    args=parser.parse_args()
    jobs=[]
    if args.stage=='pipeline':
        equivalence=ROOT/'manifests/phase03_teacher_training_equivalence.json'
        if not equivalence.exists() or json.loads(equivalence.read_text(encoding='utf-8'))['status']!='PASS':
            jobs.append(('teacher_training_equivalence',[ROOT/'scripts/phase03_teacher_replay_check.py']))
        jobs.append(('exact_development',[ROOT/'scripts/phase03_exact_development.py']))
    if args.stage in ['pipeline','after-dev','eval-dev']:
        for scene in ['coffee_martini','cut_roasted_beef']:
            jobs.append((f'eval_dev_{scene}',[ROOT/'scripts/phase03_evaluate.py','--run',ROOT/'runs/B0/dev'/scene/'seed0_loader_exact','--scene',scene,'--role','validation']))
    if args.stage in ['pipeline','after-dev','dense']:
        jobs.append(('dense_coffee',[ROOT/'scripts/phase03_dense_init.py','--protocol','final','--scene','coffee_martini',
            '--colmap',ROOT/'tools/colmap38/COLMAP-3.8-windows-cuda/COLMAP.bat','--mode','dense']))
    if args.stage in ['pipeline','after-dev','after-dense','formal-teachers']:
        for mode in ['raft','sam']:
            jobs.append((f'teacher_final_{mode}',[ROOT/'scripts/phase03_teacher_final.py',mode]))
    if args.stage in ['pipeline','after-dev','after-dense','formal-train']:
        dense=ROOT/'manifests/phase03_dense_final_coffee_martini.json'
        if args.stage in ['formal-train','after-dense']:assert dense.exists(),'Dense initialization prerequisite missing'
        cloud=str(ROOT/'data/dynerf/final/train/coffee_martini/points3D_dense_phase03.ply')
        jobs.append(('formal_coffee_dense',[ROOT/'working/MoRGS/train_phase03.py','--config',ROOT/'configs/phase03/final_coffee_martini.yaml',
            '-m',ROOT/'runs/B0/final/coffee_martini/seed0_dense','--phase03_initial_cloud',cloud]))
    if args.stage in ['pipeline','after-dev','after-dense','eval-final']:
        jobs.append(('eval_final_coffee',[ROOT/'scripts/phase03_evaluate.py','--run',ROOT/'runs/B0/final/coffee_martini/seed0_dense',
            '--protocol','final','--scene','coffee_martini','--role','test']))
    for name,arguments in jobs:
        log=ROOT/'logs'/f'phase03_batch_{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        command=[str(RUNNER)]+list(map(str,arguments))
        start=time.perf_counter()
        record=dict(name=name,command=command,started=datetime.now().astimezone().isoformat(),log=str(log))
        print('JOB_START '+json.dumps(record),flush=True)
        with log.open('w',encoding='utf-8') as handle:
            result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,cwd=ROOT)
        record.update(exit_code=result.returncode,seconds=time.perf_counter()-start,
                      finished=datetime.now().astimezone().isoformat())
        with (ROOT/'manifests/phase03_commands.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(record)+'\n')
        print('JOB_END '+json.dumps(record),flush=True)
        assert result.returncode==0,(name,result.returncode,str(log))

if __name__=='__main__':main()
