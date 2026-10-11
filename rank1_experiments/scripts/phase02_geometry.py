"""Train-only calibrated SIFT triangulation and independent epipolar controls."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
from phase02_prepare_data import ROOT,DEV,sha,save
WORK=ROOT/'working/MoRGS';sys.path.insert(0,str(WORK))
def calibration(rows,w=676,h=507):
    poses=rows[:,:15].reshape(-1,3,5);result=[]
    for p in poses:
        rot=np.stack([p[:,1],p[:,0],-p[:,2]],1)
        t=-rot.T@p[:,3]
        # Same focal convention as pinned MoRGS, using first pose's focal.
        f=poses[0,2,4]*w/poses[0,1,4]
        k=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]])
        ext=np.c_[rot.T,t];result.append((k,ext,k@ext,p[:,3]))
    return result
def epipolar(a,b,uv1,uv2):
    k1,e1,_,_=a;k2,e2,_,_=b
    r=e2[:,:3]@e1[:,:3].T;t=e2[:,3]-r@e1[:,3]
    cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    f=np.linalg.inv(k2).T@cross@r@np.linalg.inv(k1)
    x=np.c_[uv1,np.ones(len(uv1))];y=np.c_[uv2,np.ones(len(uv2))]
    fx=x@f.T;fy=y@f
    return np.abs((y*fx).sum(1))/np.sqrt(fx[:,0]**2+fx[:,1]**2+fy[:,0]**2+fy[:,1]**2+1e-20)
def write_ply(p,xyz,rgb):
    from plyfile import PlyData,PlyElement
    a=np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(['x','y','z']):a[k]=xyz[:,i]
    for k in ['nx','ny','nz']:a[k]=0
    for i,k in enumerate(['red','green','blue']):a[k]=rgb[:,i]
    PlyData([PlyElement.describe(a,'vertex')]).write(p)
def run(scene,protocol):
    source=ROOT/f'data/dynerf/{protocol}/train/{scene}'
    meta=json.loads((source/'dataset_meta.json').read_text(encoding='utf-8'));cams=meta['cameras']
    assert 'cam00' not in cams and (protocol!='dev' or 'cam01' not in cams)
    full=np.load(Path('E:/4DGS/data')/scene/'poses_bounds.npy')
    rows=full[meta['source_pose_rows']];assert np.array_equal(rows,np.load(source/'poses_bounds.npy'))
    cals=calibration(rows); features=[];images=[];reads=[]
    sift=cv2.SIFT_create(nfeatures=7000,contrastThreshold=0.02)
    for cam in cams:
        while not (ROOT/f'manifests/extraction/{scene}/{cam}.json').exists():time.sleep(2)
        p=source/cam/'0000.png';im=cv2.imread(str(p));assert im is not None
        if im.shape[:2]!=(507,676):im=cv2.resize(im,(676,507),interpolation=cv2.INTER_AREA)
        k,d=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
        features.append((np.array([x.pt for x in k]),d));images.append(im)
        reads.append(dict(path=str(p),sha256=sha(p),frame=0,camera=cam))
    centers=np.array([c[3] for c in cals]);dist=np.linalg.norm(centers[:,None]-centers[None],axis=-1)
    pairs=set()
    for i in range(len(cams)):
        for j in np.argsort(dist[i])[1:6]:pairs.add(tuple(sorted([i,int(j)])))
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=64))
    xyzs=[];rgbs=[];stats=[];examples=[]
    for i,j in sorted(pairs):
        uv1,d1=features[i];uv2,d2=features[j]
        match=matcher.knnMatch(d1,d2,k=2)
        good=[a for a,b in match if a.distance<0.75*b.distance]
        if len(good)<12:continue
        u=uv1[[x.queryIdx for x in good]];v=uv2[[x.trainIdx for x in good]]
        epi=epipolar(cals[i],cals[j],u,v)
        wrong=epipolar(cals[(i+1)%len(cams)],cals[(j+1)%len(cams)],u,v)
        p1=cals[i][2];p2=cals[j][2]
        x=cv2.triangulatePoints(p1,p2,u.T,v.T).T
        with np.errstate(divide='ignore',invalid='ignore'):x=x[:,:3]/x[:,3:]
        hom=np.c_[x,np.ones(len(x))];q1=hom@p1.T;q2=hom@p2.T
        e1=np.linalg.norm(q1[:,:2]/q1[:,2:]-u,axis=1);e2=np.linalg.norm(q2[:,:2]/q2[:,2:]-v,axis=1)
        ray1=x-centers[i];ray2=x-centers[j]
        cosine=(ray1*ray2).sum(1)/(np.linalg.norm(ray1,axis=1)*np.linalg.norm(ray2,axis=1)+1e-20)
        angle=np.arccos(np.clip(cosine,-1,1))*180/np.pi
        valid=np.isfinite(x).all(1)&(q1[:,2]>0)&(q2[:,2]>0)&(e1<1.5)&(e2<1.5)&(angle>1)&(epi<1)
        idx=np.flatnonzero(valid)
        xy=np.rint(u[idx]).astype(int);xy[:,0]=xy[:,0].clip(0,675);xy[:,1]=xy[:,1].clip(0,506)
        xyzs.append(x[idx]);rgbs.append(images[i][xy[:,1],xy[:,0],::-1])
        stats.append(dict(cameras=[cams[i],cams[j]],ratio_matches=len(good),triangulated=len(idx),
           epipolar_median=float(np.median(epi)),wrong_row_epipolar_median=float(np.median(wrong)),
           correct_epipolar_inlier_fraction=float(np.mean(epi<1)),wrong_epipolar_inlier_fraction=float(np.mean(wrong<1)),
           accepted_reprojection_median=float(np.median(np.r_[e1[idx],e2[idx]])) if len(idx) else None))
        if len(idx)>100:examples.append((len(idx),i,j,u[idx],v[idx],q1[idx,:2]/q1[idx,2:],q2[idx,:2]/q2[idx,2:]))
    xyz=np.concatenate(xyzs);rgb=np.concatenate(rgbs);assert len(xyz)>1000
    # Fixed relative voxel deduplication, no held-out image or model metric selection.
    span=np.median(np.linalg.norm(centers-centers.mean(0),axis=1));voxel=span*0.001
    cells=np.floor(xyz/voxel).astype(np.int64);_,ids=np.unique(cells,axis=0,return_index=True)
    xyz=xyz[ids];rgb=rgb[ids];assert len(xyz)>1000
    ply=source/'points3D_downsample2.ply';write_ply(ply,xyz,rgb)
    correct=np.median([s['correct_epipolar_inlier_fraction'] for s in stats]);wrong=np.median([s['wrong_epipolar_inlier_fraction'] for s in stats])
    assert correct>0.5 and correct>wrong*2,(scene,protocol,correct,wrong)
    output=dict(status='PASS',scene=scene,protocol=protocol,points=len(xyz),voxel_world_units=voxel,
      ply_path=str(ply),ply_sha256=sha(ply),initialization='Calibrated sparse SIFT triangulation, not author dense COLMAP pipeline',
      accessed_rgb=reads,public_calibration=str(Path('E:/4DGS/data')/scene/'poses_bounds.npy'),
      test_rgb_accesses=0,validation_rgb_accesses=0 if protocol=='dev' else 'cam01 is training in final protocol',
      correct_epipolar_median_inlier_fraction=float(correct),wrong_row_control_median_inlier_fraction=float(wrong),
      coordinate_convention='LLFF stored down/right/back -> CV right/down/front; w2c=R.T,-R.T*C',
      distortion='No coefficients in poses_bounds; inherited release processing. No new undistortion applied.',
      b0_vs_half_pixel_principal_delta_dev_pixels=0.125,pairs=stats)
    save(ROOT/f'manifests/geometry_{protocol}_{scene}.json',output)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    example=max(examples,key=lambda x:x[0]);_,i,j,u,v,q1,q2=example
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    for ax,img,obs,proj,label in [(axs[0],images[i],u,q1,cams[i]),(axs[1],images[j],v,q2,cams[j])]:
        ax.imshow(img[:,:,::-1]);ax.scatter(obs[::8,0],obs[::8,1],s=15,facecolors='none',edgecolors='lime');ax.scatter(proj[::8,0],proj[::8,1],s=3,c='red')
        ax.set_title(label+' observed green / projected red');ax.axis('off')
    axs[2].scatter(centers[:,0],centers[:,2],c='blue');axs[2].scatter(xyz[::20,0],xyz[::20,2],s=1,c='grey')
    for cam,c in zip(cams,centers):axs[2].text(c[0],c[2],cam,fontsize=6)
    axs[2].set_title(f'{protocol}: {len(xyz)} train-only points');axs[2].set_aspect('equal')
    fig.tight_layout();out=ROOT/f'figures/phase02/{scene}_{protocol}_reprojection.png';out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=130);plt.close(fig)
    print(json.dumps({k:output[k] for k in ['status','scene','protocol','points','correct_epipolar_median_inlier_fraction','wrong_row_control_median_inlier_fraction']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=DEV);p.add_argument('--protocol',choices=['dev','final'],default='dev');a=p.parse_args()
    cv2.setNumThreads(2);cv2.setRNGSeed(0)
    for scene in ([a.scene] if a.scene else DEV):run(scene,a.protocol)
