"""Finalize only after all declared execution gates and real evaluations pass."""
import ast
import hashlib
import json
import subprocess
import sys
import difflib
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def main():
    subprocess.run([sys.executable,str(ROOT/'scripts/phase03_status.py')],check=True)
    state=json.loads((ROOT/'experiment_state.json').read_text(encoding='utf-8'))
    assert state['phase03']=='PASS','Do not finalize incomplete work'
    subprocess.run([sys.executable,str(ROOT/'scripts/phase03_figures.py')],check=True)
    reference=ROOT/'scripts/phase02_teacher_cache.py'
    if reference.exists():
        delta=''.join(difflib.unified_diff(reference.read_text(encoding='utf-8').splitlines(True),
            (ROOT/'scripts/phase03_teacher_final.py').read_text(encoding='utf-8').splitlines(True),
            fromfile='phase02_teacher_cache.py',tofile='phase03_teacher_final.py'))
        (ROOT/'manifests/phase03_formal_teacher.patch').write_text(delta,encoding='utf-8')
    scripts=list((ROOT/'scripts').glob('phase03*.py'))+[ROOT/'scripts/protocol_cache.py',ROOT/'working/MoRGS/train_phase03.py']
    for path in scripts:ast.parse(path.read_text(encoding='utf-8'))
    artifacts=scripts+list((ROOT/'configs/phase03').glob('*.yaml'))
    artifacts+=[p for p in (ROOT/'manifests').glob('phase03*.json') if p.name!='phase03_artifact_lock.json']
    artifacts+=list((ROOT/'manifests').glob('phase03*.patch'))
    artifacts += [ROOT/'eval_protocol.json',ROOT/'resource_budget.json',ROOT/'manifests/phase03_commands.jsonl']
    for protocol in ['dev_phase03','final']:
        artifacts+=list((ROOT/'cache/teachers'/protocol).rglob('*.json'))
    for run in state['phase03_runs']:
        run_path=Path(run['run'])
        training_data=ROOT/'data/dynerf'/run['protocol']/'train'/run['scene']
        artifacts.append(training_data/('points3D_dense_phase03.ply' if run['protocol']=='final' else 'points3D_downsample2.ply'))
        for role in ['train','validation' if run['protocol']=='dev' else 'test']:
            data=ROOT/'data/dynerf'/run['protocol']/role/run['scene']
            artifacts += [data/'poses_bounds.npy',data/'dataset_meta.json']
        artifacts+=list((run_path/'snapshots').glob('*.pt'))
        artifacts += [p for p in [run_path/'cfg_args',run_path/'resources.jsonl',run_path/'training_metrics.json',
                        run_path/'initial_reuse.json',run_path/'evaluation'/('validation' if run['protocol']=='dev' else 'test')/'results.json',
                        run_path/'evaluation'/('validation' if run['protocol']=='dev' else 'test')/'per_frame.json'] if p.exists()]
    lock={str(path.relative_to(ROOT)):sha(path) for path in artifacts}
    (ROOT/'manifests/phase03_artifact_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    index=[]
    for run in state['phase03_runs']:
        root=Path(run['run'])
        for path in sorted((root/'checkpoints').glob('*.pt')):
            frame=run['expected_frames'] if path.stem=='latest' else int(path.stem.split('_')[1])
            command=[str(ROOT/'scripts/run_official.cmd'),str(ROOT/'working/MoRGS/train_phase03.py'),
                     '--config',str(ROOT/'configs/phase03'/f"{run['protocol']}_{run['scene']}.yaml"),
                     '-m',run['run'],'--start_checkpoint',str(path)]
            if run['protocol']=='final':
                command += ['--phase03_initial_cloud',str(ROOT/'data/dynerf/final/train'/run['scene']/'points3D_dense_phase03.ply')]
            index.append(dict(run=run['run'],checkpoint=str(path),frame=frame,bytes=path.stat().st_size,
                              sha256=sha(path),resume_command=command if frame<run['expected_frames'] else None,
                              complete_boundary=frame==run['expected_frames']))
        run['disk_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    (ROOT/'checkpoint_index.json').write_text(json.dumps(dict(format='phase03_frame_boundary_v1',
        recovery='Use the exact per-checkpoint resume_command; completed boundaries are for rendering. Resuming an earlier checkpoint will overwrite later frame artifacts in the selected output.',
        initialization_reuse='For a new method, fork into a separate output and retain/copy the initial prefix artifacts and resource provenance. Never write method updates into the frozen B0 output. Initial state is frame_0001.pt of the same backbone/protocol/seed.',
        boundary_only=True,checkpoints=index),indent=2),encoding='utf-8')
    state['updated_at']=datetime.now().astimezone().isoformat()
    state['phase03_finished']=state['updated_at']
    state['active_artifact_lock']='manifests/phase03_artifact_lock.json'
    state['phase03_report']='reports/phase03_baseline.md'
    for path in (ROOT/'manifests').glob('phase03*.json'):
        state['manifests'][path.name]=sha(path)
    (ROOT/'experiment_state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')
    readme='''# Rank1 MoRGS research execution

Phase03 execution gate PASS: loader-exact teacher replay, independent evaluation,
frame-boundary resume and resource profiling; dev90 coffee_martini/cut_roasted_beef,
formal300 coffee_martini at1352x1014 with train-only COLMAP dense initialization.

Read reports/phase03_baseline.md, experiment_state.json, eval_protocol.json,
resource_budget.json, checkpoint_index.json and manifests/phase03_artifact_lock.json.
Actual successful commands and failure logs: manifests/phase03_commands.jsonl.

This is B0 compatibility under explicit data/initialization/Windows differences;
paper-number reproduction is not claimed. B1/B2/M are not implemented yet.
Phase04 is next: shared projection/candidates and minimal mixture loss.
Keep original authors' source snapshot, phase01/02 records and old diagnostic runs.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    resume='''Continue only when asked to execute the next phase. Current Phase03 execution gate PASS.
Read E:/4D_Gaussian/rank1_experiment_plan_20261010/共同约束与数学定义.md and 完整分阶段实验Prompts.md.
Read rank1_experiments/reports/phase03_baseline.md and experiment_state.json.
Current dev B0 runs are seed0_loader_exact; seed0 and old smoke *_v2 are diagnostics using obsolete GPU normalization teacher caches.
Use cache/teachers/dev_phase03 and final (CPU ToTensor division before CUDA), envs/morgs-cu121, scripts/run_official.cmd.
Original author source and native required patches retained. train_phase03.py is a generated compatibility entry point; original loss block SHA is verified.
Full checkpoints are frame-boundary pickles with actual modules/optimizer, previous attrs, offset, mapping, masks and RNG. Do not use legacy capture/restore as equivalent.
Preserve B0 author offset optimizer disconnection after prune; any correction must be a separate shared control.
Formal dense uses public fixed poses + COLMAP triangulation/MVS and NumPy voxel averaging; not a claim of identical author SfM initialization.
Teacher live/replay labels/loss and gradients checked; snapshot and native renderer checked. Normal repeat-run variation is measured, not bitwise deterministic.
No new supervision or visibility/mixture module implemented. Stage04 must keep B0 reference separate from shared geometry/teacher corrections.
Do not rerun old phase01/02 finalizers or reset existing outputs.
'''
    (ROOT/'RESUME_PROMPT.txt').write_text(resume,encoding='utf-8')
    budget=json.loads((ROOT/'resource_budget.json').read_text(encoding='utf-8'))
    budget['disk_by_run']=[dict(run=r['run'],bytes=r['disk_bytes']) for r in state['phase03_runs']]
    (ROOT/'resource_budget.json').write_text(json.dumps(budget,indent=2,ensure_ascii=False),encoding='utf-8')
    # Budget updated last, refresh its lock.
    lock['resource_budget.json']=sha(ROOT/'resource_budget.json')
    (ROOT/'manifests/phase03_artifact_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    state['manifests']['phase03_artifact_lock.json']=sha(ROOT/'manifests/phase03_artifact_lock.json')
    (ROOT/'experiment_state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(status='PASS',checkpoints=len(index),script_parse='PASS',paper_number_reproduction=False)))

if __name__=='__main__':main()
