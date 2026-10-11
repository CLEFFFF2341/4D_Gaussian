"""Extend the pinned dev cache script to one full-size 300-frame scene."""
import ast
import difflib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
original=(ROOT/'scripts/phase02_teacher_cache.py').read_text(encoding='utf-8')
text=original
changes={
    'torch.from_numpy(a).permute(2,0,1).float().cuda()/255':'torch.from_numpy(a).permute(2,0,1).float().div(255).cuda()',
    "CACHE=ROOT/'cache/teachers/dev'":"DEV=['coffee_martini']\nCACHE=ROOT/'cache/teachers/final'",
    'data/dynerf/dev/train/':'data/dynerf/final/train/',
    "manifests/teacher_manifest.json":"manifests/phase03_teacher_final.json",
    "protocol='dev90'":"protocol='final300'",
    "formal300_cache='NOT_GENERATED'":"formal300_cache='IN_PROGRESS'",
    'range(89)':'range(299)',
    'range(10,91,10)':'range(10,301,10)',
    '[0,44,88]':'[0,149,298]',
    '[507,676]':'[1014,1352]',
    'sf[:,1]*=507/506;sr[:,1]*=507/506':'sf[:,1]*=1;sr[:,1]*=1',
    'fref=f*2;bref=backward*2':'fref=f;bref=backward',
    'y*=507/506 after upstream odd-height resize; input normalization unchanged':'No odd-height resize correction at 1014px; input normalization unchanged',
}
for old,new in changes.items():
    assert old in text,old
    text=text.replace(old,new)
ast.parse(text)
(ROOT/'scripts/phase03_teacher_final.py').write_text(text,encoding='utf-8')
(ROOT/'manifests/phase03_formal_teacher.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(True),text.splitlines(True),fromfile='phase02_teacher_cache.py',
    tofile='phase03_teacher_final.py')),encoding='utf-8')
print('Formal300 teacher adapter generated; coffee_martini only; pinned teachers unchanged.')
dev=original.replace("CACHE=ROOT/'cache/teachers/dev'","CACHE=ROOT/'cache/teachers/dev_phase03'")
dev=dev.replace('manifests/teacher_manifest.json','manifests/phase03_teacher_dev.json')
dev=dev.replace('torch.from_numpy(a).permute(2,0,1).float().cuda()/255',
                'torch.from_numpy(a).permute(2,0,1).float().div(255).cuda()')
dev=dev.replace("input_range='RGB float32 [0,1], upstream MoRGS behavior'",
                "input_range='RGB uint8 converted to float32 /255 on CPU, then CUDA; exact training ToTensor path'")
ast.parse(dev)
(ROOT/'scripts/phase03_teacher_dev.py').write_text(dev,encoding='utf-8')
(ROOT/'manifests/phase03_dev_teacher.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(True),dev.splitlines(True),fromfile='phase02_teacher_cache.py',
    tofile='phase03_teacher_dev.py')),encoding='utf-8')
