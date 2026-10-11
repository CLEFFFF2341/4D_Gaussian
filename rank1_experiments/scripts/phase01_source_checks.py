"""Execute the upstream argument parser and selected geometry properties on CPU."""
import ast
import hashlib
import importlib.util
import json
import sys
import zipfile
from types import SimpleNamespace
from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'sources'/'MoRGS-5c62ae88b8fb2e3d4213329dd1ec3a61c66f8902'

def save(name, value):
    (ROOT/'manifests'/name).write_text(json.dumps(value, indent=2), encoding='utf-8')

def arguments_check():
    spec=importlib.util.spec_from_file_location('upstream_arguments', SOURCE/'arguments/__init__.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg=yaml.safe_load((SOURCE/'configs/dynerf.yaml').read_text(encoding='utf-8'))
    parser=ArgumentParser()
    groups={}
    for name, cls in [('model_params','ModelParams'),('opt_params_initial','OptimizationParamsInitial'),
                      ('opt_params_rest','OptimizationParamsRest'),('pipe_params','PipelineParams'),('quantize_params','QuantizeParams')]:
        groups[name]=getattr(module,cls)(parser,cfg.get(name,{}))
    args=parser.parse_args([])
    resolved={name:vars(group.extract(args)) for name,group in groups.items()}
    save('resolved_config.json',dict(source=str(SOURCE), args=vars(args),groups=resolved,
         runtime_initial_iterations=10000, runtime_rest_iterations='8 * actual_train_camera_count',
         actual_flow_loss_coefficient=0.01, note='Executed upstream parser only. Full train.py has not been run.'))
    inventory=json.loads((ROOT/'manifests/dataset_inventory.json').read_text())
    mappings=[]
    camera_tree=ast.parse((SOURCE/'utils/camera_utils.py').read_text(encoding='utf-8'))
    camera_functions=[n for n in camera_tree.body if isinstance(n,ast.FunctionDef) and n.name in ['loadSequentialCam','sequentialCameraList_from_camInfos']]
    camera_namespace={'torch':torch,'SequentialCameraInfo':object,'SequentialCamera':lambda **kwargs: SimpleNamespace(**kwargs)}
    exec(compile(ast.Module(body=camera_functions,type_ignores=[]),'upstream_camera_functions','exec'),camera_namespace)
    for scene in inventory['scenes']:
        names=[c['camera'] for c in scene['cameras']]
        train=[name for i,name in enumerate(names) if i not in args.test_indices]
        camera_infos=[SimpleNamespace(uid=i,R=None,T=None,FovX=1.,FovY=1.,width=64,height=48,znear=.1,zfar=100.,principle_point=None) for i in range(len(names)) if i not in args.test_indices]
        runtime=camera_namespace['sequentialCameraList_from_camInfos'](camera_infos,1.,args)
        assert [c.uid for c in runtime]==list(range(len(train)))
        selected=[train[c.uid] for c in runtime if c.uid in args.flow_views]
        mappings.append(dict(scene=scene['scene'], pose_row_to_camera=dict(enumerate(names)), runtime_train_uid_to_camera=dict(enumerate(train)),
                            selected_geometry_cameras=selected,
                            teacher_peek_cameras=[train[i] for i in args.flow_views],
                            same_camera_pairing=[a==b for a,b in zip(selected,[train[i] for i in args.flow_views])],
                            proposed_validation=min(name for name in train if name not in selected),
                            mapping_status='Executed upstream camera-list and camera-loader functions with metadata-only CPU stubs; reprojection pending'))
    save('camera_mapping_audit.json', mappings)
    print('Upstream parser and camera mapping audit saved.')

def properties_check():
    tree=ast.parse((SOURCE/'scene/gaussian_model.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
    selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_resi','_xyz_fixed']]
    minimal=ast.Module(body=[ast.ClassDef(name='Probe',bases=[],keywords=[],body=selected,decorator_list=[])],type_ignores=[])
    ast.fix_missing_locations(minimal)
    namespace={}
    exec(compile(minimal, str(SOURCE/'scene/gaussian_model.py'),'exec'),namespace)
    probe=namespace['Probe']()
    decoded=torch.tensor([[1.,0.,5.]],dtype=torch.float64,requires_grad=True)
    offset=torch.tensor([[.2,0.,0.]],dtype=torch.float64,requires_grad=True)
    confidence=torch.tensor([[.5]],dtype=torch.float64,requires_grad=True)
    probe.latent_decoders={'xyz':lambda x:x}
    probe._latents={'xyz':decoded}
    probe.prev_atts={'xyz':torch.tensor([[0.,0.,5.]],dtype=torch.float64)}
    probe.xyz_before=probe.prev_atts['xyz']
    probe.mapping=torch.tensor([0])
    probe.get_motion_label=confidence
    probe.offset_3d=offset
    actual=probe._xyz_fixed()
    resi=probe._resi
    delta=actual-probe.xyz_before
    save('geometry_property_probe.json', dict(mode='upstream AST properties with identity decoder on CPU; no rasterizer',
         decoded=decoded.tolist(), offset=offset.tolist(), confidence=confidence.tolist(),
         actual_rgb_xyz=actual.tolist(), renderer_residual=resi.tolist(), actual_rgb_displacement=delta.tolist(),
         difference=(delta-resi).tolist(), rgb_offset_gradient=torch.autograd.grad(actual.sum(),offset,retain_graph=True)[0].tolist(),
         residual_offset_gradient=torch.autograd.grad(resi.sum(),offset,allow_unused=True)[0],
         note='Residual ignores offset directly; native flow also depends on actual geometry, so full flow-loss offset gradient need not be zero.'))
    print('Executed upstream geometry properties: delta != residual when offset is nonzero.')

def lock_source():
    files=[]
    archive=ROOT/'sources/morgs-5c62ae88.zip'
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            relative=Path(*Path(info.filename).parts[1:])
            data=z.read(info)
            path=SOURCE/relative
            digest=hashlib.sha256(data).hexdigest()
            assert path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest
            files.append(dict(path=relative.as_posix(),bytes=len(data),sha256=digest))
    upstream=json.loads((ROOT/'manifests/upstream_commit.json').read_text(encoding='utf-8'))
    save('source_lock.json',dict(repository='https://github.com/LeeWonJoon9868/MoRGS',
        pinned_commit='5c62ae88b8fb2e3d4213329dd1ec3a61c66f8902',upstream_main=upstream['sha'],
        acquisition='official codeload commit archive; two Git transport attempts failed', git_checkout=False,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),source_path=str(SOURCE),
        submodules='Three extension directories are vendored in the archive; no root .gitmodules. Per-file hashes lock contents.',
        native_prebuilt_note='Bundled Linux .so files are not usable Windows binaries.',
        original_archive_files_verified=True, generated_pycache_excluded=True,
        root_license_present=any((SOURCE/n).exists() for n in ['LICENSE','LICENSE.md','LICENSE.txt']),files=files))
    print('Source lock:', len(files),'files')

if __name__=='__main__':
    arguments_check()
    properties_check()
    lock_source()
