"""Real trained source state: exact CPU alpha*T on fixed sampled pixels."""
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'working/MoRGS')]
from phase04.native import snapshot, raster_candidates
from phase04.shared_flow import project, flow_loss, resolution_scale
from phase03_evaluate import RenderSnapshot
from scene.dataset_readers import readCamerasFromPoseBounds
from utils.camera_utils import loadSequentialCam
from protocol_cache import TeacherCache


def exact_pixel(xy, conic, indices, pixel):
    x, y = pixel
    trans, weights, rows = 1., [], []
    for row in indices:
        dx, dy = xy[row]-[x,y]
        a,b,c,opacity = conic[row]
        power = -.5*(a*dx*dx+c*dy*dy)-b*dx*dy
        if power > 0: continue
        alpha = min(.99, opacity*math.exp(power))
        if alpha < 1/255: continue
        next_trans = trans*(1-alpha)
        if next_trans < .0001: break
        weights.append(alpha*trans); rows.append(int(row)); trans = next_trans
    order = np.argsort(-np.asarray(weights), kind='stable')
    return np.asarray(rows)[order], np.asarray(weights)[order]


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    records, precision = [], []
    for scene in ('coffee_martini', 'cut_roasted_beef'):
        run = ROOT/'runs/phase04/frozen'/scene/'M'
        previous = RenderSnapshot(torch.load(run/'snapshots/0003.pt', weights_only=False))
        current = RenderSnapshot(torch.load(run/'snapshots/0004.pt', weights_only=False))
        uid = torch.load(run/'checkpoints/frame_0002.pt', weights_only=False)['gaussian_state']['identity'].uid
        source = snapshot(previous, uid)
        data = ROOT/'data/dynerf/dev/train'/scene
        meta = json.loads((data/'dataset_meta.json').read_text())
        infos,_ = readCamerasFromPoseBounds(str(data/'poses_bounds.npy'), meta['size_wh'], N_video_views=10)
        split = json.loads((ROOT/'manifests/splits_dev.json').read_text())['scenes'][scene]
        cache = TeacherCache(ROOT/'cache/teachers/dev_phase03',scene,split['flow_cameras'],'shared')
        for index,name in zip(split['flow_runtime_indices'],split['flow_cameras']):
            camera = loadSequentialCam(SimpleNamespace(data_device='cuda',znear=5.,zfar=300.),index,infos[index],1.)
            out = raster_candidates(source,camera,uid,64,debug=True)
            xy,conic,point_list,ranges = [v.cpu().numpy() for v in out['debug']]
            xy, conic = xy.astype(np.float64), conic.astype(np.float64)
            projected,_ = project(source['xyz'],camera.full_proj_transform,camera.world_view_transform,camera.image_width,camera.image_height)
            target,_ = project(current.get_xyz,camera.full_proj_transform,camera.world_view_transform,camera.image_width,camera.image_height)
            visible = out['ids'][out['ids']>=0].unique()
            projection_error = (projected[visible]-out['debug'][0][visible]).abs()
            precision.append(dict(scene=scene,camera=name,max_pixel_error=float(projection_error.max()),
                                   mean_pixel_error=float(projection_error.mean()),tf32_matmul=True))
            assert float(projection_error.max()) < .01, precision[-1]
            f = (target-projected).cpu().numpy()
            teacher = cache.pair(name,2,3,'cpu')
            mask = teacher['valid'].numpy()
            valid = np.flatnonzero(mask)
            magnitude = np.linalg.norm(teacher['flow'].numpy(),axis=0).reshape(-1)
            rng = np.random.default_rng(404)
            # Half deterministic uniform valid samples, half largest teacher
            # displacement. Choice depends only on frozen teacher, never method.
            pixels = np.unique(np.concatenate((rng.choice(valid,16,replace=False),
                                               valid[np.argsort(magnitude[valid])[-16:]])))
            gpu_ids, gpu_weights = out['ids'].cpu().numpy(), out['weights'].cpu().numpy()
            s = resolution_scale(camera.image_width,camera.image_height)
            for p in pixels:
                y,x = divmod(int(p),camera.image_width)
                tile = (y//16)*((camera.image_width+15)//16)+x//16
                lo,hi = ranges[tile]
                ids,weights = exact_pixel(xy,conic,point_list[lo:hi],(x,y))
                if not len(ids): continue
                actual_ids = gpu_ids[y,x][gpu_ids[y,x]>=0]
                actual_weights = gpu_weights[y,x][:len(actual_ids)]
                assert np.array_equal(ids[:64], actual_ids)
                assert np.max(abs(weights[:64]-actual_weights)) < 2e-6
                observation = teacher['flow'][:,y,x].double()[None]
                motion = torch.from_numpy(f[ids].astype(np.float64))[None]
                prior = torch.from_numpy(weights)[None]
                likelihood = -((motion[0]/s-observation[0]/s)**2).sum(-1)/(2*2.**2)+prior[0].log()
                best_index = int(likelihood.argmax())
                values = {}
                for method in ('B1','B2','H','M'):
                    full,_ = flow_loss(motion,prior,observation,torch.tensor([True]),method,scale=s,reference_double=True)
                    truncated,_ = flow_loss(motion[:,:64],prior[:,:64],observation,torch.tensor([True]),method,scale=s,reference_double=True)
                    values[method] = dict(exact=float(full),k64=float(truncated),absolute_difference=float(abs(full-truncated)))
                records.append(dict(scene=scene,camera=name,pixel=[x,y],exact_contributors=len(ids),
                                    retained_mass=float(weights[:64].sum()/weights.sum()),
                                    strongest_inlier_component_retained=best_index<64,
                                    teacher_magnitude=float(magnitude[p]),losses=values))
            del out,projected,target,visible,projection_error
            gc.collect();torch.cuda.empty_cache()
        del previous,current,source
        gc.collect();torch.cuda.empty_cache()
    assert all(r['strongest_inlier_component_retained'] for r in records)
    report=dict(status='PASS',pixels=records,projection_precision=precision,
                sample_rule='seed404 16 uniform valid +16 largest frozen teacher displacement per camera; trained frozen M 0002->0003',
                common_k=64,scope='Checks current short sequence; no guarantee for all later frames')
    (ROOT/'manifests/phase04_exact_pixels.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',pixels=len(records),min_mass=min(r['retained_mass'] for r in records),
        projection_max=max(r['max_pixel_error'] for r in precision),missed_best=0)),flush=True)


if __name__=='__main__':main()
