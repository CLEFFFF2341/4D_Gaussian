"""Freeze the seed0 search before validation regions or pilot scores exist."""
import ast
import difflib
import json
import shutil
import sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase05.common import SCENES,METHODS,read,write,sha,now,state_update

def main():
    assert not (ROOT/'manifests/phase05_protocol.json').exists(),'Frozen protocol already exists; do not rerun prepare'
    assert read(ROOT/'experiment_state.json')['phase04']=='PASS'
    lock=read(ROOT/'manifests/phase04_artifact_lock.json')
    assert all(sha(ROOT/p)==h for p,h in lock.items()),'Frozen phase04 artifact changed'
    source=ROOT/'working/MoRGS/train_phase04.py';old=source.read_text(encoding='utf-8')
    text=old.replace('from phase04.training import SharedTraining','from phase05.training import PilotTraining as SharedTraining')
    needle='            if render_flow:\n                loss += shared.flow(viewpoint_cam, recon_loss, iteration)'
    assert text.count(needle)==1
    text=text.replace(needle,'            shared.camera_context(viewpoint_cam)\n            shared.observe(viewpoint_cam, recon_loss, iteration)\n'+needle)
    needle='    args = parser.parse_args(sys.argv[1:])';assert text.count(needle)==1
    text=text.replace(needle,"    parser.add_argument('--pilot_calibrate', action='store_true')\n    parser.add_argument('--pilot_calibration_file', default='')\n"+needle)
    ast.parse(text)
    (ROOT/'working/MoRGS/train_phase05.py').write_text(text,encoding='utf-8')
    (ROOT/'patches/phase05_training_adapter.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='train_phase04.py (frozen)',tofile='train_phase05.py')),encoding='utf-8')
    frozen=ROOT/'configs/frozen_pilot_configs';frozen.mkdir(parents=True,exist_ok=True)
    initial={};jobs=[]
    for scene in SCENES:
        p=ROOT/'runs/B0/dev'/scene/'seed0_loader_exact/checkpoints/frame_0001.pt'
        initial[scene]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
        split=read(ROOT/'manifests/splits_dev.json')['scenes'][scene]
        assert split['validation']==['cam01'] and split['test']==['cam00']
        for cam in split['flow_cameras']:
            for i in range(89):
                pair=ROOT/'cache/teachers/dev_phase03'/scene/cam/f'{i:04d}_{i+1:04d}.npz'
                assert pair.exists() and pair.with_suffix('.json').exists(),pair
        data=ROOT/'data/dynerf/dev/validation'/scene
        assert read(data/'dataset_meta.json')['n_frames']==90
        assert all((data/'cam01'/f'{i:04d}.png').exists() for i in range(90))
        for method in (*METHODS,'F0','CALIBRATE'):
            ratios=(.1,.03,.3) if method in METHODS else (0.,)
            for ratio in ratios:
                trial=f'r{round(ratio*1000):03d}'
                role='search' if method in METHODS else 'diagnostic' if method=='F0' else 'calibration'
                run=ROOT/'runs/pilot'/role/scene/method/trial
                cfg=yaml.safe_load((ROOT/'configs/phase03'/f'dev_{scene}.yaml').read_text())
                cfg['model_params']['model_path']=str(run)
                path=frozen/f'{scene}_{method}_{trial}.yaml'
                path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
                jobs.append(dict(scene=scene,method=method,ratio=ratio,sigma=2.,k=64,trial=trial,role=role,
                                 config=str(path),config_sha256=sha(path),run=str(run),initial=initial[scene]))
    protocol=dict(status='FROZEN',frozen_at=now(),stage='05',scope='seed0 90-frame pilot search; conditional 300-frame three-seed repeat is a separate pending step',
          scenes=list(SCENES),methods=list(METHODS),ratios_in_order=[.1,.03,.3],sigma=2.,epsilon=.02,k=64,
          groups_per_family=3,max_groups_per_family=6,extra_sigma_search='Not invoked for this predeclared first search; no method gets additional search',
          seed=0,frames=90,resolution_wh=[676,507],density='native; same rules, no new point budget',
          initial_checkpoints=initial,calibration='All four unweighted flow gradients on the SAME pre-first-flow-backward frame2 training state; per-scene lambda=ratio*rgb_norm/flow_norm fixed per run',
          sampling='restore seed0 checkpoint camera RNG; record and require equal SHA for every complete frame',
          validation='cam01 only, never cam00; freeze teacher-only regions before scoring any pilot model',
          metrics=dict(primary='arithmetic mean of per-frame RGB PSNR, scenes equal macro weight; same FP32/SSIM/VGG LPIPS as phase03',
                       selection='one ratio per family by two-scene macro whole-frame PSNR; ties <=1e-6 use lower macro LPIPS then smaller ratio; regions cannot choose different winner',
                       region_psnr='RGB MSE restricted to region per frame, then arithmetic frame mean; empty masks reported, no invented scores',
                       region_ssim='MoRGS full-image 11x11 SSIM map, average channel then average fixed region pixels'),
          region=dict(teacher='same frozen SEA-RAFT weights/config; independent evaluation process',motion_threshold_reference_px=1.5,
                      reference_wh=[1352,1014],boundary_radius_reference_px=6,boundary_radius_dev_px=3,
                      dynamic='forward/backward valid AND magnitude in reference units >1.5',
                      boundary='morphological gradient of dynamic binary mask, radius3dev, intersect fixed FB valid',
                      static='FB valid AND NOT dynamic',teacher_inconsistent='NOT FB valid; noise/occlusion proxy, not true teacher error',
                      frame_direction='frame0..88 outgoing forward; frame89 outgoing backward to88; all masks live on their own source image grid'),
          screening=dict(strongest_control='B1/B2/H with the best independently selected macro parameters; compare all B2 and H explicitly',
                         whole='M exceeds per-scene strongest control by >=0.05dB in at least one scene and is >=-0.05dB in the other',
                         regional='Alternative: both boundary gains >=0.10dB, both whole differences >=-0.05dB',
                         lpips='No scene deterioration >0.002',repeat_threshold='Three-seed overall macro gain >=0.10dB OR boundary macro>=0.30dB with whole>=-0.05dB; LPIPS not systematically worse',
                         threshold_scope='conservative seed0 screening signal only; cannot pass final repeat gate from one seed'),
          diagnosis='If no screening signal: exactly one read-only round using existing gradient, teacher consistency, outlier/UID/K logs plus frozen phase04 direction/projection tests. No selective retraining or new modules',
          b0='read-only existing seed0_loader_exact 90-frame baseline; evaluate into runs/pilot/references, original artifacts unchanged',
          f0='one full90 run per scene, explicit flow_lambda=0; not a tuned rival',
          expansion='Only after signal, freeze selected method parameters globally and run M plus two strongest controls over both300-frame scenes at formal or one common feasible resolution, seeds0/1/2. Requires validation data and shared-head formal resource probe; never label phase05 complete before this gate',
          source_sha=read(ROOT/'experiment_state.json')['source_sha'],phase04_artifact_lock_sha256=sha(ROOT/'manifests/phase04_artifact_lock.json'),
          immutable_sources={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'phase04').glob('*.py'))+list((ROOT/'phase04').glob('*.cu'))+[source]},
          jobs=jobs,estimated_seed0_gpu_hours=[14,22],estimated_seed0_disk_gib=[160,300],free_disk_gib_at_freeze=shutil.disk_usage(ROOT).free/1024**3)
    path=ROOT/'manifests/phase05_protocol.json'
    assert not path.exists(),'Never overwrite a frozen protocol; create a new explicit version'
    write(path,protocol)
    write(ROOT/'manifests/phase05_start_state.json',read(ROOT/'experiment_state.json'))
    state_update(phase='05',status='RUNNING',phase05='RUNNING',next_phase='05',continue_gate_passed=False,
                 phase05_started=now(),phase05_protocol_sha256=sha(path),phase05_report='reports/phase05_go_no_go.md',
                 scope='Phase05 seed0 equal-budget pilot in progress; no effectiveness conclusion',
                 next_actions=['Freeze validation proxy masks','Calibrate from common training checkpoint','Run all24 native-density90-frame candidates, then evaluate and screen','Signal requires three-seed300-frame repeat before stage06'])
    print(json.dumps(dict(status='PREPARED',search_jobs=24,initial_checkpoints=initial,protocol_sha256=sha(path))))

if __name__=='__main__':main()
