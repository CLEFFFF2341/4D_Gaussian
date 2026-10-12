"""MoRGS online backbone with six attributes and an RGB-only objective."""
import argparse,json,time,math
from pathlib import Path
import torch
import common,engine
from utils.loss_utils import l1_loss,ssim
from PIL import Image
HERE=Path(__file__).resolve().parent

@torch.no_grad()
def evaluate(pc,data,frame):
    pred=engine.render(data.valcam,pc,torch.zeros(3,device='cuda'))['render'].clamp(0,1)
    gt=data.gt(frame)
    return dict(PSNR=float(-10*torch.log10((pred-gt).square().mean().clamp_min(1e-30))),
                SSIM=float(ssim(pred[None],gt[None])))

def bootstrap(scene,seed,steps,out):
    common.seed_all(seed);cfg,model,quant,opt=common.configuration(scene,seed)
    data=common.Dataset(scene);pc=common.AuditedModel(model.sh_degree,quant,model)
    pc.create_from_pcd(common.fetchPly(str(data.path/'points3D_downsample2.ply')),data.extent)
    pc.training_setup(opt);pc.initialize_ids();images=data.images(1)
    bg=torch.zeros(3,device='cuda');opt.iterations=steps
    for iteration,i in enumerate(common.camera_order(seed,1,len(images),steps),1):
        pc.update_learning_rate(iteration,quant)
        if iteration%1000==0:pc.oneupSHdegree()
        pkg=engine.render(data.cameras[i],pc,bg)
        loss=(1-opt.lambda_dssim)*l1_loss(pkg['render'],images[i])+opt.lambda_dssim*(1-ssim(pkg['render'],images[i]))
        loss.backward()
        with torch.no_grad():
            if iteration<opt.densify_until_iter:
                pc.accumulate(pkg)
                if iteration>opt.densify_from_iter and iteration%opt.densification_interval==0:
                    threshold=opt.size_threshold if iteration>opt.opacity_reset_interval*len(images) else None
                    pc.densify_and_prune(opt.densify_grad_threshold,opt.min_opacity,data.extent,threshold)
                if iteration%opt.opacity_reset_interval==0:pc.reset_opacity()
            pc.optimizer.step();pc.optimizer.zero_grad(set_to_none=True)
        if iteration%250==0 or iteration==steps:
            print(json.dumps(dict(frame=1,step=iteration,points=len(pc.uid),loss=float(loss))),flush=True)
    common.commit(pc);pc.assert_identity()
    path=out/'frame_0001.pt';common.save(path,pc,1,seed,dict(initialization_steps=steps))
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return path

def run(scene,seed,frames,out,initialization):
    common.seed_all(seed);cfg,model,quant,opt=common.configuration(scene,seed)
    data=common.Dataset(scene);pc,ck=common.load(initialization,opt)
    start_frame=ck['frame']+1;bg=torch.zeros(3,device='cuda')
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(dict(scene=scene,seed=seed,frames=frames,config=cfg,
                         initialization=str(initialization)),indent=2),encoding='utf-8')
    for frame in range(start_frame,frames+1):
        torch.cuda.synchronize();start=time.perf_counter();images=data.images(frame)
        pc.origin=torch.arange(len(pc.uid),device='cuda')
        with torch.no_grad():
            opt.set_params(frame);opt.iterations=8*len(images);pc.frame_idx=frame
            pc.update_residuals();pc.training_setup(opt)
        for iteration,i in enumerate(common.camera_order(seed,frame,len(images),opt.iterations),1):
            pc.current_step=iteration;pc.update_learning_rate(iteration,quant)
            pkg=engine.render(data.cameras[i],pc,bg)
            loss=(1-opt.lambda_dssim)*l1_loss(pkg['render'],images[i])+opt.lambda_dssim*(1-ssim(pkg['render'],images[i]))
            loss.backward()
            with torch.no_grad():
                if iteration<=math.ceil(opt.densify_until_epoch*opt.iterations) and iteration>opt.calc_dense_stats*len(images):
                    pc.accumulate(pkg)
                    if iteration%(opt.densification_interval*len(images))==0:
                        pc.densify_dynamic(opt.densify_grad_threshold,opt.min_opacity,data.extent,opt.size_threshold)
                pc.optimizer.step();pc.optimizer.zero_grad(set_to_none=True)
        pc.assert_identity();common.commit(pc)
        common.save(out/'latest.pt',pc,frame,seed,dict(boundary_committed=True,steps=opt.iterations))
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        row=dict(frame=frame,points=len(pc.uid),steps=opt.iterations,wall_seconds=elapsed,
                 loss=float(loss),metrics=evaluate(pc,data,frame))
        with (out/'frames.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=['coffee_martini','cut_roasted_beef'],required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--frames',type=int,default=90)
    p.add_argument('--bootstrap',action='store_true');p.add_argument('--init-steps',type=int,default=10000)
    p.add_argument('--initialization',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.bootstrap:
        out=a.output or HERE/'bootstrap'/a.scene/f'seed{a.seed}'
        bootstrap(a.scene,a.seed,a.init_steps,out)
    else:
        init=a.initialization or HERE/'bootstrap'/a.scene/f'seed{a.seed}/frame_0001.pt'
        out=a.output or HERE/'runs'/a.scene/f'seed{a.seed}'
        run(a.scene,a.seed,a.frames,out,init)

if __name__=='__main__':main()
