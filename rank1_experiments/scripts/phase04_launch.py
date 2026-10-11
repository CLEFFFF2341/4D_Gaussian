"""Execute an actual phase04 command, retaining failures and provenance."""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    label, script, *arguments = sys.argv[1:]
    pause = ROOT/'manifests/phase04_pause.lock'
    if label.startswith('short_') and pause.exists():
        print('Waiting between complete jobs for the phase04 reference audit', flush=True)
        while pause.exists():
            time.sleep(1)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log = ROOT/'logs'/f'phase04_{label}_{stamp}.log'
    command = [str(ROOT/'scripts/run_phase04.cmd'), str(Path(script).resolve()), *arguments]
    started = datetime.now().astimezone().isoformat()
    code_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in list((ROOT/'phase04').glob('*.py'))+list((ROOT/'phase04').glob('*.cu'))+
                   [ROOT/'working/MoRGS/train_phase04.py'] if p.exists()}
    git_commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT).decode().strip()
    t = time.perf_counter()
    print(subprocess.list2cmdline(command), flush=True)
    with log.open('w', encoding='utf-8') as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    record = dict(label=label, command=command, command_text=subprocess.list2cmdline(command),
                  started=started, ended=datetime.now().astimezone().isoformat(),
                  seconds=time.perf_counter()-t, returncode=result.returncode,
                  log=str(log.relative_to(ROOT)), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    record.update(git_commit=git_commit, code_sha256_at_start=code_hashes)
    with (ROOT/'manifests/phase04_commands.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False)+'\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)
    print(log.read_text(encoding='utf-8', errors='replace')[-10000:], flush=True)
    sys.exit(result.returncode)
