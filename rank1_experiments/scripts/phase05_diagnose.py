"""One predeclared read-only diagnosis on selected90-frame M, no new searches."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase05.common import read,write
screen=read(ROOT/'manifests/phase05_screen.json');assert not screen['signal']
trials=[read(p) for p in (ROOT/'manifests/phase05_trials').glob('*.json')]
regions=read(ROOT/'manifests/phase05_regions.json');rows=[]
for scene in ('coffee_martini','cut_roasted_beef'):
    selected=next(r for r in trials if r['method']=='M' and r['scene']==scene and r['ratio']==screen['winners']['M']['ratio'])
    run=ROOT/selected['run'];flow=[json.loads(l) for l in (run/'flow_diagnostics.jsonl').read_text().splitlines()]
    gate=[json.loads(l) for l in (run/'motion_gate_diagnostics.jsonl').read_text().splitlines()]
    mask=[r for r in regions['files'] if r['scene']==scene]
    def fraction(predicate):return sum(predicate(r) for r in flow)/max(1,len(flow))
    rows.append(dict(scene=scene,flow_events=len(flow),
                     outlier_posterior_max=max(r['outlier_posterior'] for r in flow),
                     outlier_dominated_event_fraction=fraction(lambda r:r['outlier_posterior']>.9),
                     negligible_geometry_gradient_event_fraction=fraction(lambda r:r['flow_xyz_grad_norm']<=1e-12),
                     uid_mass_min=min(r['uid_mass_retention'] for r in flow),topk_mass_min=min(r['topk_mass_retention'] for r in flow),
                     candidate_coverage_min=min(r['candidate_coverage'] for r in flow),
                     motion_gate_connected_events=sum(r['norm'] is not None for r in gate),motion_gate_events=len(gate),
                     fixed_teacher_inconsistent_fraction=sum(r['pixels']['teacher_inconsistent'] for r in mask)/(90*507*676)))
write(ROOT/'manifests/phase05_diagnosis.json',dict(status='READONLY_COMPLETE',rounds=1,scenes=rows,
     direction_projection='Frozen phase04 source-grid, rotated camera, pureZ, finite-difference and native pixel tests remain PASS; no new final90 exact per-pixel theorem claimed',
     teacher_noise='Only fixed FB inconsistency is observed; reliable-flow mask and EPE cannot certify3D truth',
     modifications=[],conclusion='No demonstrated common implementation correction from this diagnostic; retain negative result and stop scaling. No gate/trajectory or selective retry introduced.'))
print('PHASE05_READONLY_DIAGNOSIS_COMPLETE')
