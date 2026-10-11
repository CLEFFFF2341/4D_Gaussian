"""Read-only check of stage03's immutable artifact lock during phase04."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
lockpath = ROOT/'manifests/phase03_artifact_lock.json'
lock = json.loads(lockpath.read_text())
checked, deferred, errors = [], [], []
for name, expected in lock.items():
    path = ROOT/name
    if not path.exists():
        errors.append(dict(file=name, error='missing')); continue
    if path.suffix == '.pt':
        deferred.append(dict(file=name, bytes=path.stat().st_size, expected_sha256=expected))
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected: errors.append(dict(file=name, expected=expected, actual=actual))
    checked.append(name)
report = dict(status='PASS' if not errors else 'FAIL', phase03_lock_sha256=hashlib.sha256(lockpath.read_bytes()).hexdigest(),
              files_hashed=len(checked), large_pt_files_present_not_rehashed=len(deferred), errors=errors,
              scope='Full hashes of non-PT locked artifacts; presence and size of PT only; phase04 never writes old B0 runs/caches')
(ROOT/'manifests/phase04_baseline_preservation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
assert not errors
