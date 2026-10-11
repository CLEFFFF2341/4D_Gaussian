"""Seal tested startup code/calibration and predict sampling from the real checkpoint."""
import ast
import hashlib
import json
import random
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'working/MoRGS')]
from phase05.common import read,write,sha,protocol,now

p=protocol();check=read(ROOT/'manifests/phase05_startup_check.json')
assert check['status']=='PASS' and read(ROOT/'manifests/phase05_tests.json')['status']=='PASS'
assert all(sha(ROOT/file)==digest for file,digest in read(ROOT/'manifests/phase04_artifact_lock.json').items())
calibration={}
for scene in p['scenes']:
    cal=next(j for j in p['jobs'] if j['scene']==scene and j['role']=='calibration')
    file=Path(cal['run'])/'calibration.json'
    assert sha(file)==next(r['calibration_sha256'] for r in check['scenes'] if r['scene']==scene)
    calibration[str(file.relative_to(ROOT))]=sha(file)
    initial=torch.load(cal['initial']['path'],weights_only=False,map_location='cpu')
    g=random.Random();g.setstate(initial['camera_rng']);del initial
    n=len(read(ROOT/'manifests/splits_dev.json')['scenes'][scene]['train'])
    samplers={}
    for frame in range(2,91):
        stack=[];sequence=[]
        for iteration in range(1,n*8+1):
            if not stack:stack=list(range(n))
            sequence.append((iteration,stack.pop(g.randint(0,len(stack)-1))))
        samplers[frame]=hashlib.sha256(json.dumps(sequence).encode()).hexdigest()
    actual=[json.loads(l) for l in (Path(cal['run'])/'training_metrics.jsonl').read_text().splitlines()][0]
    assert samplers[2]==actual['sampler_sha256'],'Prediction must match actual calibration camera sequence'
    path=ROOT/'manifests'/f'phase05_sampler_{scene}.json';write(path,samplers)
    calibration[str(path.relative_to(ROOT))]=sha(path)
write(ROOT/'manifests/phase05_calibration_lock.json',calibration)
sources=list((ROOT/'phase05').glob('*.py'))+list((ROOT/'scripts').glob('phase05*.py'))+[ROOT/'working/MoRGS/train_phase05.py']
for path in sources:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
code={str(path.relative_to(ROOT)):sha(path) for path in sources}
write(ROOT/'manifests/phase05_code_lock.json',code)
checkpoints=[]
for directory in ('runs/pilot/calibration','runs/pilot/startup','runs/pilot/startup_v3'):
    for path in (ROOT/directory).glob('**/checkpoints/*.pt'):
        checkpoints.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path),role=directory))
write(ROOT/'manifests/phase05_startup_checkpoint_index.json',checkpoints)
paths=sources+list((ROOT/'configs/frozen_pilot_configs').glob('*.yaml'))
paths+=[ROOT/'phase05/README.md']
paths+=[ROOT/'manifests'/name for name in ('phase05_protocol.json','phase05_code_lock.json','phase05_calibration_lock.json',
        'phase05_startup_check.json','phase05_startup_checkpoint_index.json','phase05_tests.json',
        'phase05_sampler_coffee_martini.json','phase05_sampler_cut_roasted_beef.json')]
paths+=[ROOT/name for name in calibration]
write(ROOT/'manifests/phase05_startup_artifact_lock.json',{str(path.relative_to(ROOT)):sha(path) for path in set(paths)})
state=read(ROOT/'experiment_state.json')
state.update(phase05_startup='PASS',active_artifact_lock='manifests/phase05_startup_artifact_lock.json',updated_at=now())
state['manifests']['phase05_protocol.json']=sha(ROOT/'manifests/phase05_protocol.json')
state['manifests']['phase05_startup_artifact_lock.json']=sha(ROOT/'manifests/phase05_startup_artifact_lock.json')
write(ROOT/'experiment_state.json',state)
print(json.dumps(dict(status='STARTUP_SEALED',source_files=len(code),checkpoint_hashes=len(checkpoints),calibration_and_sampler_files=len(calibration))))
