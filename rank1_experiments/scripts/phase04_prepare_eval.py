"""Same frozen evaluator; redirect only its global audit outputs to phase04."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/'scripts/phase03_evaluate.py').read_text(encoding='utf-8')
text = source.replace("sys.path.insert(0, str(ROOT/'working/MoRGS'))", "sys.path.insert(0, str(ROOT/'working/MoRGS'))\nsys.path.insert(0, str(ROOT))")
assert text != source
text = text.replace("ROOT/'manifests/phase03_eval_selftest.json'", "ROOT/'manifests/phase04_eval_selftest.json'")
text = text.replace("ROOT/'eval_protocol.json'", "ROOT/'manifests/phase04_eval_protocol.json'")
ast.parse(text)
path = ROOT/'scripts/phase04_evaluate.py'
path.write_text(text, encoding='utf-8')
print('Evaluator derivative '+hashlib.sha256(path.read_bytes()).hexdigest())
