"""Verify the pinned author archive and preserved baseline reconstruction block."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
lock=json.loads((ROOT/'manifests/source_lock.json').read_text(encoding='utf-8'))
source=Path(lock['source_path'])
errors=[]
for item in lock['files']:
    path=source/item['path']
    if not path.exists() or path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        errors.append(item['path'])
assert not errors,errors
def block(path):
    text=path.read_text(encoding='utf-8')
    start=text.index('            # Compute main reconstruction losses')
    end=text.index('            if enable_debug:',start)
    return text[start:end]
original=block(ROOT/'working/MoRGS/train.py')
adapter=block(ROOT/'working/MoRGS/train_phase03.py')
assert original==adapter
result=dict(status='PASS',pinned_commit=lock['pinned_commit'],pristine_files=len(lock['files']),
            pristine_differences=errors,baseline_loss_block_sha256=hashlib.sha256(adapter.encode()).hexdigest(),
            check='Full pinned author archive file hashes, plus original versus phase03 loss block.')
(ROOT/'manifests/phase03_source_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
