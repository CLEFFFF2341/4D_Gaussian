"""COLMAP MVS initialization with frozen public calibration and train frame0 only.

Follows the author-linked dense/voxel workflow. Fixed calibration replaces SfM
remapping so the cloud stays in MoRGS coordinates and held-out RGB stays excluded.
"""
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData,PlyElement

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',choices=['dev','final'],required=True)
    p.add_argument('--scene',required=True)
    p.add_argument('--colmap',type=Path,required=True)
    p.add_argument('--mode',choices=['features','geometry','dense'],required=True)
    args=p.parse_args()
    colmap_root=args.colmap.parent if args.colmap.suffix.lower()=='.bat' else args.colmap.parent.parent
    executable=colmap_root/'bin/colmap.exe'
    environment=os.environ.copy()
    environment['PATH']=str(colmap_root/'lib')+os.pathsep+environment['PATH']
    environment['QT_PLUGIN_PATH']=str(colmap_root/'lib/plugins')
    data=ROOT/'data/dynerf'/args.protocol/'train'/args.scene
    # Development dense geometry also uses 1352x1014 frame0, same training camera set.
    images_source=ROOT/'data/dynerf/final/train'/args.scene
    split=json.loads((ROOT/'manifests'/f'splits_{args.protocol}.json').read_text(encoding='utf-8'))['scenes'][args.scene]
    cameras=split['train']
    assert 'cam00' not in cameras
    if args.protocol=='dev':assert 'cam01' not in cameras
    work=ROOT/'initialization/colmap_dense'/args.protocol/args.scene
    images=work/'images'
    images.mkdir(parents=True,exist_ok=True)
    accessed=[]
    for camera in cameras:
        source=images_source/camera/'0000.png'
        destination=images/f'{camera}.png'
        if not destination.exists():shutil.copy2(source,destination)
        accessed.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    poses=np.load(data/'poses_bounds.npy')[:,:-2].reshape(-1,3,5)
    W,H=1352,1014
    focal=float(poses[0,2,4])/(float(poses[0,1,4])/W)
    cx,cy=(W-1)/2,(H-1)/2
    db=work/'database.db'
    commands=[]
    started=time.perf_counter()
    def run(name,*arguments):
        command=[str(executable),name]+[str(x) for x in arguments]
        commands.append(command)
        with (work/f'{name}.log').open('a',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=colmap_root,env=environment)
        assert result.returncode==0,(name,result.returncode,str(work/f'{name}.log'))
        print('COLMAP_DONE',name,flush=True)
    if args.mode=='features':
        run('feature_extractor','--database_path',db,'--image_path',images,
            '--ImageReader.camera_model','PINHOLE','--ImageReader.single_camera','1',
            '--ImageReader.camera_params',f'{focal},{focal},{cx},{cy}',
            '--SiftExtraction.max_image_size','4096','--SiftExtraction.max_num_features','16384',
            '--SiftExtraction.estimate_affine_shape','1','--SiftExtraction.domain_size_pooling','1',
            '--SiftExtraction.use_gpu','0','--SiftExtraction.num_threads','8')
        run('exhaustive_matcher','--database_path',db,'--SiftMatching.use_gpu','0','--SiftMatching.num_threads','8')
    else:
        known=work/'known'
        known.mkdir(exist_ok=True)
        (known/'cameras.txt').write_text(f'1 PINHOLE {W} {H} {focal} {focal} {cx} {cy}\n',encoding='utf-8')
        (known/'points3D.txt').write_text('',encoding='utf-8')
        with sqlite3.connect(db) as connection:
            identifiers={name:(image_id,camera_id) for image_id,name,camera_id in connection.execute('SELECT image_id,name,camera_id FROM images')}
        lines=[]
        for camera,pose in zip(cameras,poses):
            converted=np.concatenate([pose[:,1:2],-pose[:,0:1],pose[:,2:4]],axis=1)
            R=-converted[:,:3]
            R[:,0]*=-1
            T=-converted[:,3].dot(R)
            xyzw=Rotation.from_matrix(R.T).as_quat()
            q=xyzw[[3,0,1,2]]
            identifier,camera_id=identifiers[f'{camera}.png']
            assert camera_id==1,camera_id
            lines.append(f'{identifier} '+ ' '.join(map(str,[*q,*T]))+f' 1 {camera}.png\n\n')
        (known/'images.txt').write_text(''.join(lines),encoding='utf-8')
        sparse=work/'sparse'
        sparse.mkdir(exist_ok=True)
        run('point_triangulator','--database_path',db,'--image_path',images,
            '--input_path',known,'--output_path',sparse)
        dense=work/'dense'
        reusable_geometry=(dense/'stereo/patch-match.cfg').exists() and (dense/'sparse/images.bin').exists()
        if reusable_geometry:
            for camera in cameras:
                image=dense/'images'/f'{camera}.png'
                assert image.exists() and hashlib.sha256(image.read_bytes()).digest()==hashlib.sha256((images/f'{camera}.png').read_bytes()).digest(),camera
            print('REUSE_VERIFIED_UNDISTORTED_GEOMETRY',flush=True)
        else:
            run('image_undistorter','--image_path',images,'--input_path',sparse,
                '--output_path',dense,'--output_type','COLMAP')
        if args.mode=='geometry':
            (work/'commands_geometry.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
            return
        run('patch_match_stereo','--workspace_path',dense,'--workspace_format','COLMAP',
            '--PatchMatchStereo.geom_consistency','true','--PatchMatchStereo.gpu_index','0')
        run('stereo_fusion','--workspace_path',dense,'--workspace_format','COLMAP',
            '--input_type','geometric','--output_path',dense/'fused.ply','--StereoFusion.num_threads','8')
        points=PlyData.read(str(dense/'fused.ply'))['vertex'].data
        xyz=np.column_stack([points[k] for k in ['x','y','z']])
        colors=np.column_stack([points[k] for k in ['red','green','blue']])
        raw_count=len(xyz)
        assert raw_count>1000,raw_count
        voxel=.02
        while len(xyz)>40000:
            grid=np.floor(xyz/voxel).astype(np.int64)
            _,inverse=np.unique(grid,axis=0,return_inverse=True)
            counts=np.bincount(inverse)
            xyz=np.column_stack([np.bincount(inverse,weights=xyz[:,j])/counts for j in range(3)])
            colors=np.column_stack([np.bincount(inverse,weights=colors[:,j])/counts for j in range(3)])
            voxel+=.01
        output=data/'points3D_dense_phase03.ply'
        vertex=np.empty(len(xyz),dtype=[(k,'f4') for k in ['x','y','z','nx','ny','nz']]+[(k,'u1') for k in ['red','green','blue']])
        for j,k in enumerate(['x','y','z']):vertex[k]=xyz[:,j]
        for k in ['nx','ny','nz']:vertex[k]=0
        for j,k in enumerate(['red','green','blue']):vertex[k]=np.clip(np.round(colors[:,j]),0,255)
        PlyData([PlyElement.describe(vertex,'vertex')]).write(str(output))
        record=dict(status='PASS',protocol=args.protocol,scene=args.scene,raw_dense_points=raw_count,
            initialization_points=len(xyz),cloud=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            accessed_rgb=accessed,public_poses=str(data/'poses_bounds.npy'),seconds=time.perf_counter()-started,
            commands=commands,author_reference='hustvl/4DGaussians multipleviewprogress.sh',
            geometry_reused=reusable_geometry,
            prior_geometry_commands=json.loads((work/'commands_geometry.json').read_text(encoding='utf-8')) if reusable_geometry else [],
            differences=['Frozen public poses + point_triangulator replace mapper/re-estimated LLFF poses.',
                         'NumPy voxel averaging replaces Open3D voxel implementation; same iterative .02,+.01 up to40000 policy.',
                         'Training views only; cam01 included only in formal.'])
        (ROOT/'manifests'/f'phase03_dense_{args.protocol}_{args.scene}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record,indent=2))
    (work/f'commands_{args.mode}.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')

if __name__=='__main__':main()
