import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCENES=('coffee_martini','cut_roasted_beef')
METHODS=('B1','B2','H','M')

def now():return datetime.now().astimezone().isoformat()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def state_update(**fields):
    p=ROOT/'experiment_state.json';state=read(p)
    state.update(updated_at=now(),**fields);write(p,state)

def protocol():
    p=ROOT/'manifests/phase05_protocol.json'
    d=read(p);assert d['status']=='FROZEN'
    return d
