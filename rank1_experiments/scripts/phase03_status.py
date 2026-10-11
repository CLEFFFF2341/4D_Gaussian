"""Report actual artifacts; pending work never satisfies the phase gate."""
import ast
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def load(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

def main():
    entries=[]
    for protocol,scene,tag,frames in [('dev','coffee_martini','seed0_loader_exact',90),
                                    ('dev','cut_roasted_beef','seed0_loader_exact',90),
                                    ('final','coffee_martini','seed0_dense',300)]:
        run=ROOT/'runs/B0'/protocol/scene/tag
        path=run/'resources.jsonl'
        rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x] if path.exists() else []
        role='validation' if protocol=='dev' else 'test'
        evaluation=load(run/'evaluation'/role/'results.json')
        snapshots=list((run/'snapshots').glob('*.pt'))
        entry=dict(protocol=protocol,scene=scene,run=str(run),expected_frames=frames,
                   actual_frames=len(snapshots),training_complete=len(snapshots)==frames and (run/'training_metrics.json').exists(),
                   evaluation=evaluation,resources=rows)
        frame_ids=sorted(int(p.stem) for p in snapshots)
        entry['contiguous_frames']=frame_ids==list(range(1,frames+1))
        entry['training_complete']=entry['training_complete'] and entry['contiguous_frames']
        if evaluation:
            entry['evaluation_complete']=evaluation['frames']==frames
        else:entry['evaluation_complete']=False
        if rows:
            entry.update(training_and_save_seconds=sum(x['train_and_save_seconds'] for x in rows),
                         peak_allocated=max(x['peak_allocated'] for x in rows),
                         peak_reserved=max(x['peak_reserved'] for x in rows),
                         max_points=max(x['points'] for x in rows),
                         cpu_rss=max(x['cpu_rss'] for x in rows))
            entry['initial_training_and_save_seconds']=rows[0]['train_and_save_seconds']
            entry['shared_initial_checkpoint']=str(run/'checkpoints/frame_0001.pt')
            entry['shared_initial_saved_seconds_per_branch']=rows[0]['train_and_save_seconds']
            entry['initial_reused_from']=rows[0].get('reused_from')
            entry['disk_bytes']=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
        train_log=load(run/'training_metrics.json')
        if train_log:
            entry['update_author_frame_seconds']=sum(x['Frame time'] for x in train_log if x['Frame index']>1)
            entry['update_author_frame_seconds_definition']='Author synchronized frame timer; initial10000 excluded; includes phase03 checkpoint I/O and training-camera diagnostics, excludes offline teachers.'
        entries.append(entry)
    resume=load(ROOT/'manifests/phase03_resume_check.json')
    if resume and 'continuous_run' not in resume:
        commands=ROOT/'manifests/phase03_commands.jsonl'
        records=[json.loads(x) for x in commands.read_text(encoding='utf-8').splitlines()] if commands.exists() else []
        matches=[x for x in records if x['name']=='resume_check' and x['exit_code']==0]
        if matches:
            command=matches[-1]['command']
            resume.update(continuous_run=command[command.index('--continuous')+1],
                          resumed_run=command[command.index('--resumed')+1],
                          repeat_run=command[command.index('--repeat')+1],
                          provenance_log=matches[-1]['log'],provenance_note='Execution record supplies path metadata; numerical check not rerun.')
            (ROOT/'manifests/phase03_resume_check.json').write_text(json.dumps(resume,indent=2),encoding='utf-8')
    evaluator=load(ROOT/'manifests/phase03_eval_selftest.json')
    dense=load(ROOT/'manifests/phase03_dense_final_coffee_martini.json')
    equivalence=load(ROOT/'manifests/phase03_teacher_training_equivalence.json')
    dense_pose=load(ROOT/'manifests/phase03_dense_pose_check.json')
    loss_preserved=load(ROOT/'manifests/phase03_loss_preservation.json')
    source_verified=load(ROOT/'manifests/phase03_source_verification.json')
    final_teachers=load(ROOT/'manifests/phase03_teacher_final.json')
    cache_checked=load(ROOT/'manifests/phase03_cache_metadata_check.json')
    visual_qa=load(ROOT/'manifests/phase03_visual_qa.json')
    gate=all(x['training_complete'] and x['evaluation_complete'] for x in entries)
    gate=bool(gate and resume and resume['status']=='PASS' and evaluator and evaluator['status']=='PASS' and dense
              and equivalence and equivalence['status']=='PASS' and dense_pose and dense_pose['status']=='PASS'
              and loss_preserved and loss_preserved['status']=='PASS')
    gate=bool(gate and source_verified and source_verified['status']=='PASS'
              and final_teachers and final_teachers.get('formal300_cache')=='PASS'
              and dense and dense.get('status')=='PASS')
    gate=bool(gate and cache_checked and cache_checked['status']=='PASS')
    gate=bool(gate and visual_qa and visual_qa['status']=='PASS')
    if gate:gate=resume.get('continuous_run','').endswith('continuous_loader_exact')
    if gate:gate=all(x['peak_allocated']<22*2**30 and x['peak_reserved']<22*2**30 for x in entries)
    budget=dict(status='PASS' if gate else 'IN_PROGRESS',gpu='RTX4090 24GiB WDDM',
        runs=entries,teacher_dev=load(ROOT/'manifests/phase03_teacher_dev.json'),
        teacher_final=final_teachers,
        dense_initialization=dense,
        checks=dict(resume=resume,teacher_equivalence=equivalence,dense_pose=dense_pose,loss_preservation=loss_preserved,source_verification=source_verified,cache_extent=cache_checked,visual_qa=visual_qa),
        note='Train-and-save includes snapshots/checkpoints; teacher/init/evaluation listed separately. Reused init also counted at full standalone cost.')
    budget['disk_by_run']=[dict(run=x['run'],bytes=x.get('disk_bytes',0)) for x in entries]
    budget['teacher_cache_disk_bytes']={protocol:sum(p.stat().st_size for p in (ROOT/'cache/teachers'/protocol).rglob('*') if p.is_file())
                                       for protocol in ['dev_phase03','final']}
    command_path=ROOT/'manifests/phase03_commands.jsonl'
    command_records=[json.loads(x) for x in command_path.read_text(encoding='utf-8').splitlines()] if command_path.exists() else []
    budget['preprocessing_process_seconds_including_retries']={name:sum(x['seconds'] for x in command_records if x['name']==name)
        for name in ['dense_coffee','teacher_final_raft','teacher_final_sam']}
    budget['preprocessing_cost_note']='Includes initial serial writer attempt resumed with parallel lossless compression; raft_seconds in teacher manifest is the last attempt only. Dense CPU features/geometry preceding batch commands remain in COLMAP logs.'
    (ROOT/'resource_budget.json').write_text(json.dumps(budget,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# 阶段 03：B0 基线与实际预算', '',f'更新时间：{datetime.now().astimezone().isoformat()}。状态：'+('执行门槛通过' if gate else '执行中，未通过完整序列门槛')+'。', '',
        '本阶段保留固定 MoRGS 的损失、增点和逐帧更新，初始10000 iterations；后续8 epochs/view。缓存适配、Windows必要修复与独立评估使本地运行应标为 **B0 compatibility / 严格数据协议**，不能直接称为作者论文数字复现。', '',
        '开发两场景使用阶段02训练视角稀疏初始化；正式 coffee_martini 使用 COLMAP 稠密初始化（如果已生成），所有初始化仅使用本划分训练视角第0帧。不同初始化的指标不用于判定算法增益。', '',
        '| 运行 | 已保存帧 | 统一PSNR / SSIM / LPIPS | 训练+保存秒 | allocated / reserved GiB |',
        '|---|---:|---|---:|---|']
    for entry in entries:
        ev=entry['evaluation']
        quality='待评估' if not ev else '/'.join(f"{ev['unified_fp32'][k]:.5f}" for k in ['PSNR','SSIM','LPIPS'])
        seconds=entry.get('training_and_save_seconds',0)
        mem='待测量' if not entry['resources'] else f"{entry['peak_allocated']/2**30:.3f} / {entry['peak_reserved']/2**30:.3f}"
        lines.append(f"| {entry['protocol']} {entry['scene']} | {entry['actual_frames']}/{entry['expected_frames']} | {quality} | {seconds:.1f} | {mem} |")
    lines += ['', '渲染FPS含一次warmup、每个保留checkpoint测量5次原GaussianModel（包括latent decoder）同步渲染，加载不计入；另存decoded snapshot FPS，二者不混称。']
    for entry in entries:
        ev=entry['evaluation']
        if ev:lines.append(f"- {entry['protocol']} {entry['scene']}: full model {ev['full_model_rendering_fps']:.2f} FPS; snapshot {ev['rendering_fps']:.2f} FPS; evaluation {ev['seconds']:.1f}s.")
    lines+=['', '## 成本与复用', '',
        '| 运行 | 首帧10000迭代+保存秒 | 后续训练帧计时秒（含保存） | CPU帧边界RSS最大GiB | 当前磁盘GiB |',
        '|---|---:|---:|---:|---:|']
    for entry in entries:
        if entry['resources']:
            lines.append(f"| {entry['protocol']} {entry['scene']} | {entry['initial_training_and_save_seconds']:.1f} | {entry.get('update_author_frame_seconds',0):.1f} | {entry['cpu_rss']/2**30:.3f} | {entry['disk_bytes']/2**30:.3f} |")
    lines+=['', '首帧checkpoint是该主干seed0的分叉起点。每个方法独立从零的成本需加上首帧训练；共享初始化时每个额外分叉可节省表中首帧时间。coffee开发主运行已从同协议smoke首帧复用；其成本表仍计入首次初始化成本。每帧资源记录保存checkpoint大小，最终索引保存全部可恢复文件SHA256。', '',
        '教师和稠密初始化为可复用的预处理成本；精确秒数和显存见 resource_budget.json 的 teacher_dev、teacher_final、dense_initialization 字段。实际命令的进程墙钟时间含启动、加载与I/O，见 manifests/phase03_commands.jsonl，父流水线时间不可与其子作业重复求和。', '',
        '已有开发可视化显示场景整体对齐，但留出视角背景和边缘存在覆盖/纹理误差。稀疏与稠密初始化结果不能归为监督方法收益；其原因未由单变量对照完全隔离。此阶段的PASS仅代表所列执行检查，论文复现和方法有效性仍未成立。']
    lines+=['', '正式cam00第299帧的桌面、窗户、柜子和人物整体对齐，仍有细纹理和手部/躯干误差。正式300帧指标曲线稳定；开发cut后段PSNR下降，尚未单变量隔离内容变化、初始化与原运动更新的影响。可視化核验见 phase03_visual_qa.json。', '',
        '## 可审阅产物', '',
        '- 完整恢复索引：checkpoint_index.json；实际命令：manifests/phase03_commands.jsonl。',
        '- 逐帧数值：各run的evaluation/validation或evaluation/test/per_frame.json及resources.jsonl。',
        '- 图像与曲线：figures/phase03/final_coffee_martini_heldout.png、quality_curves.png、resource_curves.png。',
        '- 冻结文件SHA256：manifests/phase03_artifact_lock.json。']
    lines+=['', '## 评估与隔离', '',
        '训练进程不构造测试 loader，并通过图像文件打开守卫拒绝 validation/test PNG。开发验证 cam01 与正式最终测试 cam00 只在单独评估进程读取，测试数字不用于选超参数。原训练日志中 Val 字段实际是训练cam0的诊断，不能当开发验证成绩；真正成绩只看 evaluation/validation。', '',
        '统一 PSNR 用FP32 RGB MSE，SSIM沿用11×11 sigma1.5的实现，统一LPIPS以VGG0.1、RGB*2−1输入。作者命名空间另用8位量化PSNR/SSIM和原[0,1] LPIPS。相同图像、已知常量误差与LPIPS对称性自检见 eval_protocol.json / phase03_eval_selftest.json。', '',
        '所有本次运行 use_xyz_legacy=False，遵循官方CLI默认的_xyz_fixed路径。官方参数帮助另将--use_xyz_legacy标为论文数字复现入口，因此本次不满足该论文配置复现声明。此选择先于正式测试冻结，未据测试成绩切换。', '',
        '每个run内cfg_args保存执行时参数。开发运行的教师路径字段仍为原相对路径，但缓存适配器实际加载ROOT/weights固定SHA的同一教师；最终YAML将这些字段绝对化，研究超参数不变。恢复和评估以实际状态与缓存manifest为准。', '',
        '## 保存、恢复与源码边界', '',
        'checkpoint 是帧边界完整状态：Gaussian属性/解码器/offset/previous attributes/mapping/optimizer/相机缓存、采样随机状态、Python/NumPy/CPU/CUDA RNG和split generator。每帧更新latest；保留首帧、2/5/10帧、每30帧与最后一帧。逐帧FP32解码快照用于独立渲染，不能代替恢复checkpoint。', '',
        '10帧连续运行与从第5帧中断后的续跑及同checkpoint重跑已比较（状态以manifest为准）。CUDA执行下重跑也有增点和RGB差异；恢复误差必须不高于两次重跑差异的2倍，并限制平均RGB差异不超过1/255。局部最大误差仍可较大，这不是逐位确定性保证。其来源尚未隔离，不把所有差异直接归因于CUDA atomics。', '',
        '发现作者 prune_points 将 offset_3d 切片，未绑定到 _prune_optimizer 返回的替代参数；因此增点后offset可能与optimizer组脱离。B0参考保留该行为并完整保存两个对象，任何修复必须另列共享控制，不能算混合监督贡献。', '',
        '教师构造的随机状态消耗及TF32设置在训练前重现，随后卸载；训练只读固定FP32 B0缓存。共享FP16 resize/FB-valid协议未用于B0。', '',
        '阶段03核对发现：阶段02缓存的图像归一化在CUDA完成，而真实ToTensor loader在CPU完成，舍入微差可产生约0.0065px全缓存最大流差及少量SAM union像素变化。已新建 dev_phase03 缓存并冷启动重训，旧cache/teachers/dev和旧runs/.../seed0仅为工程诊断。训练入口抽查在线教师/缓存flow、heatmap、union与loss相同，xyz梯度最大差约6.5e-9。证据见 phase03_cache_input_precision.json / phase03_teacher_training_equivalence.json。', '',
        '## 初始化核验', '',
        'COLMAP工具来自官方3.8 Windows CUDA发行包，SHA256 60910e2d79539ba8825628bd147f900a5cd61646b5a9ad5f62dbec0e9b78e8d0。参考作者链接的 multipleviewprogress.sh 完成稠密深度融合与至多40000点的voxel下采样。为保持发布坐标和测试隔离，使用公开固定标定+point_triangulator，而非重跑mapper并重新估计LLFF位姿；NumPy替代Open3D voxel实现。这些差异明确保留，尚未声称点云与作者相同。', '',
        '实际初始化命令和读入图像SHA存于 phase03_dense_final_coffee_martini.json。原稀疏点云与旧研究输出保留。', '',
        '## 继续条件', '',
        '只有两开发场景90帧、一个正式300帧均能训练与独立评估，恢复检查通过并有显存/耗时实测，才允许进入阶段04。只打通训练或只保存部分帧，状态保持RUNNING。论文质量复现另需逐项核对初始化、作者配置/legacy模式及参考论文指标，不能用本报告自动替代。']
    (ROOT/'reports/phase03_baseline.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state=load(ROOT/'experiment_state.json')
    state.update(phase='03',phase03='PASS' if gate else 'RUNNING',status='PASS' if gate else 'RUNNING',
                 updated_at=datetime.now().astimezone().isoformat(),
                 scope='Phase03 B0 compatibility: loader-exact caches, isolated evaluation, resume, dev90 x2, formal dense300; paper-number reproduction separately bounded',
                 continue_gate_passed=gate,next_phase='04' if gate else '03',baseline_execution_complete=gate,
                 baseline_reproduction_complete=False,phase03_runs=[{k:v for k,v in x.items() if k!='resources'} for x in entries])
    state['baseline_quality_scope']='Execution/alignment verified; development sparse-init coverage and later cut quality decline remain limitations; paper numbers not reproduced.'
    (ROOT/'experiment_state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(gate_passed=gate,runs=[(e['scene'],e['actual_frames']) for e in entries])))

if __name__=='__main__':main()
