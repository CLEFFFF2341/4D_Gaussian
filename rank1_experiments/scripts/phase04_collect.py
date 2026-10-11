"""Audit finished short runs, derive tables and a separate checkpoint index."""
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'working/MoRGS')]
from phase03_hooks import atomic_save


def main():
    entries, checkpoints = [], []
    for topology in ('frozen','native'):
        for scene in ('coffee_martini','cut_roasted_beef'):
            for method in ('F0','B1','B2','H','M'):
                run = ROOT/'runs/phase04'/topology/scene/method
                resources = [json.loads(line) for line in (run/'resources.jsonl').read_text().splitlines()]
                assert [r['frame'] for r in resources] == [2,3,4],run
                iterations = 128 if scene=='coffee_martini' else 144
                assert all(r['iterations'] == iterations for r in resources)
                assert all(r['peak_allocated'] < 20*1024**3 for r in resources)
                cfg = json.loads((run/'resolved_config.json').read_text())
                assert cfg['flow_method']==method and cfg['flow_k']==64 and cfg['flow_sigma']==2 and cfg['flow_epsilon']==.02
                assert not cfg['use_xyz_legacy'] and cfg['test_indices']==[]
                flow = [json.loads(line) for line in (run/'flow_diagnostics.jsonl').read_text().splitlines()]
                assert {r['frame'] for r in flow} == {2,3,4}
                if method=='F0':
                    assert all(not r['flow_enabled'] and r['flow_grad_norm']==0 and r['flow_loss']==0 for r in flow)
                    common_lambda = 0.
                else:
                    common_lambda = cfg['effective_flow_lambda']
                    assert all(r['lambda_flow']==common_lambda and r['flow_xyz_grad_norm']>0 and r['flow_offset_grad_norm']>0 for r in flow)
                    assert min(r['candidate_coverage'] for r in flow) >= .9
                    assert min(r['uid_mass_retention'] for r in flow) >= .95
                    assert min(r['topk_mass_retention'] for r in flow) >= .95
                    calibrated = common_lambda*cfg['calibration_flow']['norm']/cfg['calibration_rgb']['norm']
                    assert abs(calibrated-.1) < 1e-8
                latest = torch.load(run/'checkpoints/latest.pt',weights_only=False,map_location='cpu')
                state = latest['gaussian_state']
                assert latest['frame']==4
                uid = state['identity']
                assert len(uid.uid)==resources[-1]['points'] and uid.uid.unique().numel()==len(uid.uid)
                for group in state['optimizer'].param_groups:
                    if group['name'] in state['param_names']:
                        assert group['params'][0] is state['_latents'][group['name']]
                    if group['name']=='offset_3d': assert group['params'][0] is state['offset_3d']
                assert state['offset_3d'] is not None
                if topology=='frozen':
                    assert len(set(r['points'] for r in resources))==1 and state['topology_version']==0
                    # Earlier instrumentation omitted per-frame UID files. These
                    # are exactly derivable because this control never changes topology.
                    for frame in (2,3,4):
                        path=run/'identities'/f'{frame:04d}.pt'
                        if not path.exists():
                            atomic_save(dict(uid=uid.uid,parent=uid.parent,next_uid=uid.next_uid,
                                             topology_version=0,derived_from='complete frame4 checkpoint; frozen topology; equal point counts verified'),path)
                else:
                    assert state['topology_version']>0, 'Native density must actually execute'
                    assert all((run/'identities'/f'{f:04d}.pt').exists() for f in (2,3,4))
                evaluation=json.loads((run/'evaluation/validation/results.json').read_text())
                assert evaluation['frames']==3 and evaluation['role']=='validation'
                assert all(r['abs_max']<1e-6 for r in evaluation['full_model_snapshot_checks'])
                entry=dict(topology=topology,scene=scene,method=method,k=64,frames=3,iterations_per_frame=iterations,
                    initial_checkpoint=cfg['start_checkpoint'],seed=cfg['seed'],lambda_flow=common_lambda,
                    train_and_save_seconds=sum(r['train_and_save_seconds'] for r in resources),
                    peak_allocated_gib=max(r['peak_allocated'] for r in resources)/1024**3,
                    peak_reserved_gib=max(r['peak_reserved'] for r in resources)/1024**3,
                    final_points=resources[-1]['points'],topology_version=state['topology_version'],
                    uid_mass_min=None if method=='F0' else min(r['uid_mass_retention'] for r in flow),
                    topk_mass_min=None if method=='F0' else min(r['topk_mass_retention'] for r in flow),
                    candidate_coverage_min=None if method=='F0' else min(r['candidate_coverage'] for r in flow),
                    outlier_posterior_max=None if method=='F0' else max(r['outlier_posterior'] for r in flow),
                    validation_psnr=evaluation['unified_fp32']['PSNR'],validation_ssim=evaluation['unified_fp32']['SSIM'],
                    validation_lpips=evaluation['unified_fp32']['LPIPS'],run=str(run.relative_to(ROOT)))
                entries.append(entry)
                for path in sorted((run/'checkpoints').glob('*.pt')):
                    frame=4 if path.name=='latest.pt' else int(path.stem.split('_')[-1])
                    checkpoints.append(dict(path=str(path.relative_to(ROOT)),frame=frame,bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),method=method,scene=scene,topology=topology))
                del latest,state,uid
                gc.collect()
    output=ROOT/'manifests/phase04_short_summary.json'
    for directory, role in ((ROOT/'runs/phase04/resume', 'resume_repeat_diagnostic'),
                            (ROOT/'runs/phase04_versions', 'superseded_engineering_version')):
        for path in directory.glob('**/checkpoints/*.pt'):
            checkpoint=torch.load(path,weights_only=False,map_location='cpu')
            checkpoints.append(dict(path=str(path.relative_to(ROOT)),frame=checkpoint['frame'],
                bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),role=role))
            del checkpoint;gc.collect()
    summary=dict(status='PASS',runs=entries,initialization_shared_per_scene=True,
                 actual_iterations_verified=True,offset_optimizer_identity_verified=True,
                 quality_claim='implementation smoke test only; not phase05 go/no-go or performance evidence')
    output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (ROOT/'manifests/phase04_short_table.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
    (ROOT/'manifests/phase04_checkpoint_index.json').write_text(json.dumps(checkpoints,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',runs=len(entries),checkpoints=len(checkpoints),seconds=sum(r['train_and_save_seconds'] for r in entries))))


if __name__=='__main__':main()
