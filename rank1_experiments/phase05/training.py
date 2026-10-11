"""Recording and fixed-checkpoint calibration around the frozen shared head."""
import hashlib
import json
import math
from pathlib import Path
import torch
from phase04.training import SharedTraining,motion_gradients
from phase04.native import raster_candidates,candidate_loss
from phase04.shared_flow import project,resolution_scale
from .common import ROOT,read,write,sha


class PilotTraining(SharedTraining):
    def __init__(self,args,pc):
        assert not args.flow_freeze_topology,'Pilot main comparison requires native density'
        super().__init__(args,pc)
        self.calibrating=args.pilot_calibrate
        self.did_calibrate=False
        if self.calibrating:self.enabled=True
        else:
            assert args.flow_lambda>=0,'Pilot lambda must be fixed before training'
            cfg=read(self.out/'resolved_config.json')
            cfg.update(phase='05',pilot_protocol_sha256=sha(ROOT/'manifests/phase05_protocol.json'),
                       calibration_file=args.pilot_calibration_file,
                       lambda_policy='fixed from common pre-update frame2 training state')
            write(self.out/'resolved_config.json',cfg)

    def begin_frame(self,frame):
        super().begin_frame(frame)
        self.rgb_sum=0.;self.sample_sequence=[];self.flow_rows=[]
        self.did_write=False

    def observe(self,camera,rgb_loss,iteration):
        self.rgb_sum+=float(rgb_loss.detach())
        self.sample_sequence.append((iteration,camera.uid))

    def flow(self,camera,rgb_loss,iteration):
        if self.calibrating:
            if not self.did_calibrate:
                assert self.frame==2
                c=raster_candidates(self.source,camera,self.pc.identity.uid,self.args.flow_k)
                t=self.cache.pair(self.cameras[camera.uid],0,1,'cuda')
                rgb=motion_gradients(rgb_loss,self.pc)
                assert rgb['norm']>0
                rows={}
                for method in ('B1','B2','H','M'):
                    value,diag=candidate_loss(self.source,self.pc.get_xyz,camera,c,t['flow'],t['valid'],method,2.,.02)
                    g=motion_gradients(value,self.pc)
                    assert g['norm']>1e-12 and g['xyz_norm']>0 and g['offset_norm']>0
                    rows[method]=dict(loss=float(value),gradient=g,unit_lambda=rgb['norm']/g['norm'],diagnostics=diag)
                write(self.out/'calibration.json',dict(status='PASS',frame=2,camera=camera.uid,
                      iteration=iteration,initial_checkpoint=self.args.start_checkpoint,
                      rgb=rgb,methods=rows,sigma=2.,k=64,
                      sampler_prefix_sha256=hashlib.sha256(json.dumps(self.sample_sequence).encode()).hexdigest(),
                      calibration_state='Before first flow backward; identical RGB-only prefix from seed0 B0 frame1'))
                print('PHASE05_CALIBRATION_PASS',flush=True)
                self.did_calibrate=True
            return rgb_loss.new_zeros(())
        self.did_write=False
        value=super().flow(camera,rgb_loss,iteration)
        if self.did_write and self.enabled and value.requires_grad:
            gate=self.pc._latents['motion_label']
            g=torch.autograd.grad(value/self.lambda_flow,[gate],retain_graph=True,allow_unused=True)[0]
            with (self.out/'motion_gate_diagnostics.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(frame=self.frame,iteration=iteration,camera=camera.uid,
                       norm=None if g is None else float(g.norm()),mean_confidence=float(self.pc.get_motion_label.mean())))+'\n')
        return value

    @torch.no_grad()
    def sample_diagnostics(self,camera_uid):
        # Deterministic stride sample, no RNG consumption or model-dependent masks.
        c=self.candidates[camera_uid][1];t=self.teachers[camera_uid]
        h,w=t['flow'].shape[-2:]
        loc=t['valid'].flatten().nonzero().flatten()
        loc=loc[::max(1,len(loc)//2048)][:2048]
        if not len(loc):return dict(diagnostic_samples=0)
        camera=self._camera_for_write
        sxy,sv=project(self.source['xyz'],camera.full_proj_transform,camera.world_view_transform,w,h)
        txy,tv=project(self.pc.get_xyz,camera.full_proj_transform,camera.world_view_transform,w,h)
        ids=c['ids'].flatten(0,1)[loc];rows=c['mapping'][ids.clamp_min(0)]
        q=c['weights'].flatten(0,1)[loc]*((ids>=0)&(rows>=0)&sv[ids.clamp_min(0)]&tv[rows.clamp_min(0)])
        a=q/q.sum(-1,keepdim=True).clamp_min(1e-30)
        delta=(txy[rows.clamp_min(0)]-sxy[ids.clamp_min(0)])/resolution_scale(w,h)
        y=t['flow'].flatten(1).T[loc]/resolution_scale(w,h)
        logp=math.log(.98)+torch.where(q>0,a.clamp_min(1e-30).log(),-torch.inf)-math.log(2*math.pi*self.args.flow_sigma**2)
        logp=logp-(delta-y[:,None]).square().sum(-1)/(2*self.args.flow_sigma**2)
        out=math.log(.02)-math.log(2*math.pi*128**2)-y.square().sum(-1)/(2*128**2)
        post=torch.cat((logp,out[:,None]),-1).softmax(-1)
        bins=post.new_tensor([0.,.1,.5,.9,.99,1.00001])
        histogram=torch.histogram(post[:,-1].cpu(),bins=bins.cpu())[0].tolist()
        quantiles=torch.quantile(post[:,-1],post.new_tensor([0.,.5,.9,1.])).tolist()
        mean=(a[...,None]*delta).sum(1)
        return dict(diagnostic_samples=len(loc),diagnostic_sampling='first <=2048 fixed-valid stride pixels; posterior uses common mixture for every method',
                    candidate_count_mean=float((q>0).sum(-1).float().mean()),
                    candidate_count_min=int((q>0).sum(-1).min()),
                    outlier_posterior_quantiles=quantiles,outlier_histogram=histogram,
                    outlier_histogram_edges=bins.tolist(),
                    max_component_posterior_mean=float(post.max(-1).values.mean()),
                    teacher_mean_epe_reference=float((mean-y).norm(dim=-1).mean()))

    def write(self,row):
        if self.enabled and not self.calibrating:
            row.update(self.sample_diagnostics(row['camera']))
        super().write(row)
        self.flow_rows.append(row);self.did_write=True

    def camera_context(self,camera):self._camera_for_write=camera

    def end_frame(self,frame):
        super().end_frame(frame)
        n=len(self.sample_sequence)
        expected=128 if Path(self.args.source_path).name=='coffee_martini' else 144
        assert n==expected or frame==1,(frame,n,expected)
        row=dict(frame=frame,iterations=n,rgb_loss_mean=self.rgb_sum/max(n,1),
                 sampler_sha256=hashlib.sha256(json.dumps(self.sample_sequence).encode()).hexdigest(),
                 flow_diagnostic_events=len(self.flow_rows),flow_enabled=self.enabled and not self.calibrating)
        for key in ('flow_loss','flow_grad_norm','rgb_grad_norm','outlier_posterior','responsibility_entropy',
                    candidate_count_mean','teacher_mean_epe_reference'):
            values=[d[key] for d in self.flow_rows if key in d]
            row[key+'_event_mean']=sum(values)/len(values) if values else None
        with (self.out/'training_metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if self.calibrating:assert self.did_calibrate
