"""Serialized subprocess execution with start records, logs and immutable source hashes."""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def launch(label,script,*arguments):
    log=ROOT/'logs'/f'phase05_{label}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.log'
    command=[str(ROOT/'scripts/run_phase04.cmd'),str(Path(script).resolve()),*map(str,arguments)]
    code=list((ROOT/'phase04').glob('*.py'))+list((ROOT/'phase04').glob('*.cu'))+list((ROOT/'phase05').glob('*.py'))
    code+=list((ROOT/'scripts').glob('phase05*.py'))+[ROOT/'working/MoRGS/train_phase05.py']
    record=dict(label=label,command=command,command_text=subprocess.list2cmdline(command),
                started=datetime.now().astimezone().isoformat(),log=str(log.relative_to(ROOT)),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                code_sha256_at_start={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code if p.exists()})
    with (ROOT/'manifests/phase05_started_commands.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    print(record['command_text'],flush=True);start=time.perf_counter()
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        result=p.wait()
    record.update(ended=datetime.now().astimezone().isoformat(),seconds=time.perf_counter()-start,
                  returncode=result,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    with (ROOT/'manifests/phase05_commands.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    print(log.read_text(encoding='utf-8',errors='replace')[-1800:],flush=True)
    return result

if __name__=='__main__':sys.exit(launch(*sys.argv[1:]))
