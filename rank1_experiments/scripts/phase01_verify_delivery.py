"""Verify frozen evidence without rerunning GPU or dataset experiments."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
state = json.loads((ROOT / 'experiment_state.json').read_text(encoding='utf-8'))
assert state['status'] in ['PARTIAL','PASS']
assert state['continue_gate_passed'] is (state['status']=='PASS')
assert state['training_results'] is None
for name, digest in state['manifests'].items():
    assert hashlib.sha256((ROOT / 'manifests' / name).read_bytes()).hexdigest() == digest, name
lock = json.loads((ROOT / 'manifests/source_lock.json').read_text(encoding='utf-8'))
source = Path(lock['source_path'])
for entry in lock['files']:
    assert hashlib.sha256((source / entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
assert len(lock['files']) == 3290
for script in (ROOT / 'scripts').glob('*.py'):
    ast.parse(script.read_text(encoding='utf-8-sig'), filename=str(script))
json_count = 0
for manifest in (ROOT / 'manifests').glob('*.json'):
    json.loads(manifest.read_text(encoding='utf-8-sig'))
    json_count += 1
for name in ['reports/phase01_source_audit.md', 'README.md', 'RESUME_PROMPT.txt']:
    assert (ROOT / name).stat().st_size > 0
result = dict(verified_at=datetime.now(timezone.utc).isoformat(),
              original_source_files_verified=len(lock['files']),
              manifest_json_files_parsed=json_count, frozen_state_hashes_verified=True,
              scripts_parse=True, status='PASS', phase_status=state['status'])
(ROOT / 'manifests/delivery_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
