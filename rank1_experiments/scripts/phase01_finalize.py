"""Freeze phase-01 evidence and an honest continuation state."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'manifests/environment_initial.json').exists():
    raise SystemExit('Historical initial-audit finalizer: use phase01_recovery_finalize.py for current state.')
SOURCE=ROOT/'sources/MoRGS-5c62ae88b8fb2e3d4213329dd1ec3a61c66f8902'

def read(name):
    return json.loads((ROOT/'manifests'/name).read_text(encoding='utf-8-sig'))

def write(name,value):
    (ROOT/'manifests'/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

now=datetime.now(timezone(timedelta(hours=8))).isoformat()
health=read('environment_health.json')
native=read('native_probe_mask_stride_8.json')
environment=dict(updated_at=now,os=platform.platform(),gpu='RTX 4090 24GB',
    official_target=dict(python='3.10.18',torch='2.5.1',torchvision='0.20.1',cuda='12.1'),
    official_environment_status='NOT_CREATED: package SSL EOF / HTTP timeouts',
    official_environment_attempts=['logs/conda_create.json','logs/conda_create_explicit_v2.json'],
    diagnostic_environment=dict(path=str(ROOT/'envs/morgs-native-probe'),python=sys.version,
                               system_site_packages=True,base_environment_read_only='C:/Users/CLEFFFF/anaconda3/envs/Ex4DGS',
                               torch=health['torch'],cuda=health['torch_cuda'],
                               gpu_smoke=True,rasterizer_forward_backward=True,full_training=False),
    wsl='No installed distribution reported',system_cuda_toolkit='11.8',msvc='VS 2019 Community 16.11.57',
    teachers=dict(raft_import=True,sam2_import=True,checkpoint_loaded=False,pretrained_inference=False,
                  sam2_default_config_composition=False,sam2_corrected_config_composition=True),
    device_memory_caveat='WDDM nvidia-smi usage and torch mem_get_info are not interchangeable budget measurements.',
    model_training_peak_memory=None,model_training_seconds=None,native_smoke_peak_bytes=native['peak_allocated'])
write('environment.json',environment)

teachers=[]
for name,folder,config,weight,origin in [
    ('SEA-RAFT','RAFT','RAFT/config/eval/spring-M.json','RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth',
     'https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M'),
    ('SAM2.1','sam2','sam2/configs/sam2.1/sam2.1_hiera_l.yaml','sam2/checkpoints/sam2.1_hiera_large.pt',
     'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt')]:
    tree_digest=hashlib.sha256()
    for p in sorted((SOURCE/folder).rglob('*.py')):
        tree_digest.update(p.relative_to(SOURCE).as_posix().encode())
        tree_digest.update(p.read_bytes())
    teachers.append(dict(name=name,vendored_by_morgs_commit='5c62ae88b8fb2e3d4213329dd1ec3a61c66f8902',
        python_source_tree_sha256=tree_digest.hexdigest(),config_path=config,
        config_sha256=hashlib.sha256((SOURCE/config).read_bytes()).hexdigest(),checkpoint_path=weight,
        checkpoint_exists=(SOURCE/weight).exists(),checkpoint_sha256=None,origin=origin,
        status='IMPORT_ONLY; pretrained inference not executed'))
write('teacher_dependencies.json',dict(teachers=teachers,raft_resnet_dependency='RAFT constructor calls torchvision ResNet34 ImageNet weights before loading its checkpoint.',
      sam2_c_extension='Bundled sam2/_C.so is a Linux binary; postprocessing extension requires Windows build or an explicitly recorded behavior change.',
      sam2_config_default='./sam2.1/sam2.1_hiera_l.yaml',sam2_config_resolved='configs/sam2.1/sam2.1_hiera_l.yaml'))

mapping=read('camera_mapping_audit.json')
write('data_preparation_status.json',dict(updated_at=now,
    n3dv_raw_scenes=6,raw_cameras=120,all_headers_and_samples_readable=True,
    development_strict_decode=dict(first_sweep_pass=37,first_sweep_total=38,
        failed='cut_roasted_beef/cam10.mp4',identical_archive_content=True,subsequent_single_process_passes=4,
        diagnosis='Transient decoding/concurrency behavior unresolved; do not label raw video corrupt or silently repair.'),
    prepared_for_morgs=False,points3D_downsample2_missing_all_scenes=True,
    development_validation_proposal={m['scene']:m['proposed_validation'] for m in mapping[:2]},
    test_camera='cam00',camera_reprojection_checked=False,splits_frozen=False,
    derived_data_root_proposal=str(ROOT/'data/dynerf'),
    meet_room=dict(local_found=False,downloaded=False,
        origins=['https://www.modelscope.cn/datasets/DAMOXR/dynamic_nerf_meeting_room_dataset/summary',
                 'https://drive.google.com/drive/folders/1lNmQ6_ykyKjT6UKy-SnqWoSlI5yjh3l_'],
        exact_scene_names=None,test_camera_id=None)))

manifest_names=['source_lock.json','environment.json','dataset_inventory.json','camera_mapping_audit.json',
                'resolved_config.json','teacher_dependencies.json','native_probe_pristine.json','native_probe_mask_stride_8.json']
state=dict(updated_at=now,last_audit_run_ended_at=now,start_time='2026-10-10T22:33:30+08:00',timezone='Asia/Shanghai',
    audit_elapsed_seconds=(datetime.fromisoformat(now)-datetime.fromisoformat('2026-10-10T22:33:30+08:00')).total_seconds(),
    scope='Phase 01: environment/source audit plus existing-dataset inventory',phase='01',status='PARTIAL',
    source_sha='5c62ae88b8fb2e3d4213329dd1ec3a61c66f8902',
    patch_sha=hashlib.sha256((ROOT/'patches/mask_stride_8.patch').read_bytes()).hexdigest(),
    manifests={n:hashlib.sha256((ROOT/'manifests'/n).read_bytes()).hexdigest() for n in manifest_names},
    completed=['Pinned source archive and all original file hashes','GPU tensor smoke',
               'Native rasterizer build and tiny forward/backward','Mask-stride bug reproduction and isolated correction regression',
               'Native flow scale diagnostic','Upstream argument parsing','Actual camera UID mapping',
               'Geometry-property CPU gradient probe','Teacher imports and SAM2 config composition',
               'Six-scene inventory and two-scene decoding audit','Save/restore/UID and evaluation source audit'],
    unmet=['Official Python3.10/PyTorch2.5.1/CUDA12.1 isolated environment',
           'All required custom operators and pretrained teachers exercised in that environment',
           'End-to-end training/render/evaluation/resume correctness',
           'Prepared dataset, initialization provenance and reprojection'],
    continue_gate_passed=False,phase02='NOT_STARTED',phase03='NOT_STARTED',training_results=None,
    next_actions=['Recover pinned dependency acquisition without changing the old environment',
                  'Validate all native operators and pretrained SEA-RAFT/SAM2 calls',
                  'Prepare dedicated dynerf data tree with directories only and zero-padded PNGs',
                  'Resolve source-grid/projection/offset issues under common controls',
                  'Add full restartable checkpoint and UID design before any complete baseline run'],
    recovery_prompt=str(ROOT/'RESUME_PROMPT.txt'))
(ROOT/'experiment_state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')
print('Frozen phase 01 as PARTIAL. Training gate has not passed.')
