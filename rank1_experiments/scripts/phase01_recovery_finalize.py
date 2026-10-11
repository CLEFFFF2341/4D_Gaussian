"""Freeze the repaired environment and advance only the phase-01 prerequisite gate."""
import hashlib
import json
import platform
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys
import torch
import torchvision

ROOT=Path(__file__).resolve().parents[1]
MAN=ROOT/'manifests'
NOW=datetime.now(timezone(timedelta(hours=8))).isoformat()

def read(name):
    return json.loads((MAN/name).read_text(encoding='utf-8-sig'))

def write(name,data):
    (MAN/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()

assert sys.version_info[:3]==(3,10,18)
assert torch.__version__=='2.5.1' and torchvision.__version__=='0.20.1'
assert torch.version.cuda=='12.1' and torch.cuda.is_available()
for path,backup in [(MAN/'environment.json',MAN/'environment_initial.json'),
                    (MAN/'teacher_dependencies.json',MAN/'teacher_dependencies_initial.json'),
                    (ROOT/'experiment_state.json',MAN/'experiment_state_initial.json')]:
    if not backup.exists():
        backup.write_bytes(path.read_bytes())

checks=['official_knn_smoke.json','official_raster_smoke.json','official_codec_smoke.json',
        'official_model_smoke.json','teacher_convert_smoke.json','teacher_raft_smoke.json',
        'teacher_sam_smoke.json','streaming_evaluation_smoke.json','windows_camera_paths_test.json']
for name in checks:
    assert read(name)['status']=='PASS',name
assert read('native_probe_official_mask_stride_8.json')['mask_stride_regression_passed']
assert 'No broken requirements found.' in (ROOT/'logs/official_pip_check.log').read_text(encoding='utf-8-sig')
for name,usage in [('official_train_help.log','usage: train.py'),('official_metrics_help.log','usage: metrics_video.py')]:
    assert usage in (ROOT/'logs'/name).read_text(encoding='utf-8-sig'),name

source_lock=read('source_lock.json')
source=Path(source_lock['source_path'])
work=ROOT/'working/MoRGS'
delta=[]
for entry in source_lock['files']:
    assert sha(source/entry['path'])==entry['sha256'],entry['path']
    actual=sha(work/entry['path'])
    if actual!=entry['sha256']:
        delta.append(dict(path=entry['path'],pristine_sha256=entry['sha256'],working_sha256=actual))
generated=[x for x in delta if x['path'].endswith('.pyc') or '.egg-info/' in x['path']]
code_delta=[x for x in delta if x not in generated]
assert {x['path'] for x in code_delta}=={'scene/cameras.py','submodules/gaussian-rasterization-grad/cuda_rasterizer/backward.cu'}
write('working_source_delta.json',dict(source_commit=source_lock['pinned_commit'],modified_source_files=code_delta,
    regenerated_archive_artifacts=generated,
    added_sam2_extension_source_commit='f882beb157546530a7e66a5f72ef1da4f74593d7',
    sam2_cuda_source_sha256=sha(ROOT/'sources/sam2_connected_components.cu'),
    sam2_binary_sha256=sha(work/'sam2/_C.cp310-win_amd64.pyd')))

fetch={x['fn']:x for x in json.loads((ROOT/'logs/conda_solve_offline.json').read_text(encoding='utf-8-sig'))['actions']['FETCH']}
packages=[]
for url in (MAN/'conda_explicit.txt').read_text().splitlines():
    if not url.startswith('https://'):
        continue
    filename=url.rsplit('/',1)[-1]
    archive=Path('C:/Users/CLEFFFF/anaconda3/pkgs')/filename
    digest=sha(archive)
    expected=fetch.get(filename,{}).get('sha256')
    if expected:
        assert digest==expected,filename
    packages.append(dict(url=url,filename=filename,bytes=archive.stat().st_size,sha256=digest,
                         expected_sha256=expected))
write('conda_package_lock.json',dict(packages=packages,local_recovery=True,
    acquisition_note='Original URLs retained; local explicit paths avoid failed remote cache lookup.'))

weights=[ROOT/'weights/sea_raft_model.safetensors',ROOT/'weights/Tartan-C-T-TSKH-spring540x960-M.pth',
         ROOT/'weights/sam2.1_hiera_large.pt',ROOT/'weights/resnet34-b627a593.pth',
         ROOT/'cache/torch/hub/checkpoints/vgg16-397923af.pth',ROOT/'cache/torch/hub/checkpoints/vgg.pth']
weight_records=[dict(path=str(x),bytes=x.stat().st_size,sha256=sha(x)) for x in weights]
assert weight_records[0]['sha256']=='cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102'
assert weight_records[3]['sha256'].startswith('b627a593')
assert weight_records[4]['sha256'].startswith('397923af')
write('weight_lock.json',dict(weights=weight_records,raft_conversion=read('teacher_convert_smoke.json')))
write('native_wheel_lock.json',dict(wheels=[dict(path=str(x),sha256=sha(x),bytes=x.stat().st_size)
                                          for x in sorted((ROOT/'wheels_cu121').glob('*.whl'))]))

teachers=read('teacher_dependencies_initial.json')
for teacher in teachers['teachers']:
    key='raft' if teacher['name']=='SEA-RAFT' else 'sam'
    checkpoint=ROOT/'weights'/('Tartan-C-T-TSKH-spring540x960-M.pth' if key=='raft' else 'sam2.1_hiera_large.pt')
    teacher.update(checkpoint_path=str(checkpoint),checkpoint_exists=True,checkpoint_sha256=sha(checkpoint),
                   status='PRETRAINED_MINIMUM_CALL_PASS',smoke_manifest=f'teacher_{key}_smoke.json')
teachers.update(updated_at=NOW,raft_weight_conversion='Strict model load after reconstructing shared bn3/downsample aliases; all original tensors preserved.',
                sam2_c_extension='Windows CUDA extension compiled from official pinned source; component areas 4 and 6 verified.')
write('teacher_dependencies.json',teachers)

environment=dict(updated_at=NOW,os=platform.platform(),gpu='RTX 4090 24GB',
    official_target=dict(python='3.10.18',torch='2.5.1',torchvision='0.20.1',cuda='12.1'),
    official_environment_status='CREATED_AND_MINIMUM_CALLS_VERIFIED',
    environment_path=str(ROOT/'envs/morgs-cu121'),python=sys.version,system_site_packages=False,
    toolkit=dict(path=str(ROOT/'envs/morgs-cu121'),nvcc='12.1.105',architecture='sm_89',msvc='VS2019 14.29'),
    dependency_evidence=['conda_package_lock.json','conda_official_packages.json','pip_official_freeze.txt','native_wheel_lock.json'],
    teachers=dict(pretrained_raft=True,pretrained_sam2=True,sam2_cuda=True),
    native_operators=dict(simple_knn=True,torch_cluster=True,diff_rasterizer=True,gradient_rasterizer=True,torchac=True),
    source_variant='Local mask-stride and Windows separator corrections; not an unmodified author benchmark',
    smoke_model=True,smoke_evaluation=True,full_scene_training=False,
    model_training_peak_memory=None,model_training_seconds=None,
    teacher_smoke_resources={key:read(f'teacher_{key}_smoke.json') for key in ['raft','sam']},
    preserved=['Original data and Ex4DGS environment','Pristine pinned source'],
    memory_caveat='Tiny and teacher smoke peaks do not establish full-sequence training capacity; external WDDM load remains separate.')
write('environment.json',environment)

initial=read('experiment_state_initial.json')
state=dict(initial)
state.pop('audit_elapsed_seconds',None)
state.update(updated_at=NOW,last_audit_run_ended_at=NOW,status='PASS',
    initial_audit_elapsed_seconds=initial['audit_elapsed_seconds'],
    audit_span_seconds=(datetime.fromisoformat(NOW)-datetime.fromisoformat(initial['start_time'])).total_seconds(),
    scope='Phase 01 environment/source prerequisites; full baseline reproduction remains pending',
    continue_gate_passed=True,baseline_reproduction_complete=False,
    recovery_first_logged_at=datetime.fromtimestamp((ROOT/'logs/conda_create_offline_recovery.json').stat().st_ctime,
                                                    timezone(timedelta(hours=8))).isoformat(),
    official_environment=str(ROOT/'envs/morgs-cu121'),working_source=str(work),
    phase01_gate=dict(cuda=True,native_operators=True,pretrained_teachers=True,source_lock=True,
                      configuration_and_training_entry=True,evaluation_entry=True),
    completed=initial['completed']+['Pinned official core environment recovered offline','All native operators tested in target versions',
        'Strict SEA-RAFT weight conversion and bidirectional inference','SAM2 checkpoint, CUDA postprocessing and two-frame propagation',
        'Actual GaussianModel two-frame render/backward/optimizer smoke','Upstream metrics contract and equivalent streaming adapter'],
    unmet=[],phase02='NOT_STARTED',phase03='NOT_STARTED',training_results=None,
    future_validation=['Derived two-scene data and train-only initialization/reprojection','Teacher cache quality, precision and replay checks',
        'Source-grid/flow projection and offset controls','Densification identity and arbitrary-frame checkpoint/resume',
        'Full baseline quality, training peak memory, duration and FPS'],
    next_actions=['Execute phase 02 using the repaired environment and pinned teachers',
        'Create separate dynerf data tree; freeze camera splits and original-ID teacher mapping',
        'Verify initialization provenance and reprojection before phase 03'],
    patches={x.name:sha(x) for x in sorted((ROOT/'patches').glob('*.patch'))},
    manifests={x.name:sha(x) for x in [MAN/'source_lock.json',MAN/'environment.json',MAN/'teacher_dependencies.json',
        MAN/'weight_lock.json',MAN/'working_source_delta.json',MAN/'conda_package_lock.json',MAN/'native_wheel_lock.json']
        +[MAN/name for name in checks]+[MAN/'native_probe_official_mask_stride_8.json']},
    recovery_prompt=str(ROOT/'RESUME_PROMPT.txt'))
(ROOT/'experiment_state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(phase01='PASS',full_baseline_reproduction=False,conda_packages=len(packages),
                     archive_files_verified=len(source_lock['files']),working_source_changes=len(code_delta),
                     regenerated_artifacts=len(generated)),indent=2))
