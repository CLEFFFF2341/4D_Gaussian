"""Finalize stage04 only after its concrete implementation gate passes."""
import ast
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def load(name):return json.loads((ROOT/'manifests'/name).read_text(encoding='utf-8'))


def main():
    tests, cpu, probe, exact, summary, resume, preserve = [load(name) for name in (
        'phase04_tests.json','phase04_cpu_tests.json','phase04_real_probe.json','phase04_exact_pixels.json',
        'phase04_short_summary.json','phase04_resume_check.json','phase04_baseline_preservation.json')]
    assert all(d['status']=='PASS' for d in (tests,cpu,probe,exact,summary,resume,preserve))
    assert len(summary['runs'])==20 and probe['common_k']==64
    rows=summary['runs'];flow=[r for r in rows if r['method']!='F0']
    finite=max(t['max_absolute_error'] for t in cpu['tests'] if 'max_absolute_error' in t)
    rgb=max(r['rgb_forward_max_error'] for r in probe['real_batch_results'] if r['method']=='F0')
    rgbgrad=max(r['rgb_gradient_max_error'] for r in probe['real_batch_results'] if r['method']=='F0')
    table='\n'.join(f"| {r['scene']} | {r['topology']} | {r['method']} | {r['train_and_save_seconds']:.1f} | {r['peak_allocated_gib']:.2f}/{r['peak_reserved_gib']:.2f} | {r['final_points']} | {r['lambda_flow']:.6g} |" for r in rows)
    quality='\n'.join(f"| {r['scene']} | {r['topology']} | {r['method']} | {r['validation_psnr']:.4f} | {r['validation_ssim']:.4f} | {r['validation_lpips']:.4f} |" for r in rows)
    methoderrors={m:max(p['losses'][m]['absolute_difference'] for p in exact['pixels']) for m in ('B1','B2','H','M')}
    commands=[json.loads(line) for line in (ROOT/'manifests/phase04_commands.jsonl').read_text().splitlines()]
    errors=[r for r in commands if r['returncode']!=0]
    report=f'''# 阶段04：共享候选头与最小混合目标

阶段04实现门槛 **PASS**。完成两场景、五个损失开关、冻结/原生密度两种设置共20个短训单元；每个单元从该场景相同 seed0 B0 frame1 checkpoint 分叉，完整更新 frame2—4。通过计算、梯度、RGB回归、UID、精确候选和断点恢复检查。**这不构成 M 改善重建质量的证据，阶段05尚未执行。**

## 对齐的方案与范围

完整采用 `rank1_experiment_plan_20261010/共同约束与数学定义.md` 和《完整分阶段实验Prompts.md》阶段04任务。主干仍为 MoRGS `5c62ae88b8fb2e3d4213329dd1ec3a61c66f8902`，开发 coffee_martini/cut_roasted_beef，676×507，原八轮/训练视角更新，实际每帧128/144迭代；数据 extent=90，完成三帧后正常保存并停止。初始化、相机采样 RNG、教师、RGB/运动正则和密度参数共享。没有目标可见性 gate、轨迹模块、Meet Room 或第二主干训练。

原阶段03仍为 B0 兼容协议，`use_xyz_legacy=False`，未确认论文数字复现；其稀疏开发初始化的背景质量限制仍存在。阶段04没有覆盖或改写 B0 成绩。

## 实现

- `phase04/shared_flow.py`：CPU精确贡献参考、UID、完整相机投影、单位缩放、B1/B2/H/M/F0。
- `phase04/candidates.cu`：只读解析已核验的原 MoRGS geometry/bin/image buffers，按原深度顺序与原阈值计算全贡献 αT；T 包含未匹配的源贡献者。存活 UID 过滤后按贡献取 top-K。原 RGB API、forward、backward 和已安装原生模块保持原状。
- `phase04/native.py`：冻结源帧真实 RGB XYZ/opacity/scale/rotation，目标使用当前 `pc.get_xyz`；候选权重/ID/几何有效性停止梯度，流经相机投影回传 XYZ、motion confidence、offset 的原参数链。分块计算损失，固定教师 Ω 分母，空候选按异常分布计分。
- `phase04/model.py`：独立共享子类维护 UID/parent；clone/split 子点获新 UID，prune 移除 UID，不用祖先冒充过去轨迹。共同修复原 prune 后 offset 与优化器 Parameter 断开的错误；所有共享对照，包括 F0，使用这项修复，绝不计为 M 的贡献。
- `train_phase04.py`：从冻结阶段03入口生成一个共同训练入口，只用 `--flow_method` 切换损失。F0 不调用新候选头/流损失；其 RGB 仍正常优化。checkpoint 保留完整主干状态、优化器、RNG、UID 和固定 λ；快照用于渲染，UID单独保存。

初版常数 σ=2参考像素、ε=.02、out σ=128；s=sqrt(W²+H²)/1024。FP32 log Gaussian/logsumexp，包括完整归一化常数；H 包含相同异常候选。B1 空候选使用固定异常 NLL，单位与其他损失不同，所以用梯度比例校准 λ。每次在共同初始训练过程首次流 batch 上设 flow/RGB 运动参数梯度比例 .1，随后 λ 固定；只是阶段04诊断，不是阶段05完成超参数搜索或正式跨场景冻结。正式运行可显式传 `--flow_lambda`。

## 针对性证据

CPU float64 有限差分最大绝对误差 {finite:.3g}；包括真实世界XYZ→投影→M的链与源状态 detach。CPU/GPU损失与梯度、原native缓冲贡献、K1/8/32/64、零权重、空集合、极小α、极大位移、异常主导、无效像素、重复候选、非方形 resize、前后符号、相机旋转/深度和 source-grid 足迹均通过。负 NLL 不被截断。

真实batch：B1/B2/H/M 均有非零 XYZ/实际优化器 offset 梯度；F0 流梯度为0，RGB梯度非零。读取新头前后的参数没有改变；RGB最大前向差 {rgb:.3g}，RGB梯度最大重复差 {rgbgrad:.3g}（原 CUDA 不宣称逐位确定）。真实模型 clone/split/prune 和 optimizer identity 测试两场景都通过。

解析案例包含静态、连续单表面、相反前后景、消失/新显露、透明和反射代理。相反运动示例中均值产生不属于任何候选的中间运动；M优于B2。透明平均观测反例中B2优于M。极远全部候选可被异常分量吞掉，M梯度为0。不存在“所有案例M都更好”的结论；真实教师拟合不等于3D轨迹真值。

## K与覆盖

K8在八个相同初始化视角仅保留约59%—66%贡献；K32约93%—96%，未全部达到预先设定的每相机加权贡献质量≥95%门槛。因此所有方法统一 K64：初始化视角约99.7%—99.8%。短训记录中最小 UID 保留质量 {min(r['uid_mass_min'] for r in flow):.8f}，最小 top-K 保留质量 {min(r['topk_mass_min'] for r in flow):.8f}，最小候选像素覆盖 {min(r['candidate_coverage_min'] for r in flow):.8f}。原生密度控制实际执行，完整 checkpoint 的 UID 唯一且与点数一致；未发生大面积身份丢失。

训练后两场景八视角按“seed404随机16个有效像素 + 教师位移最大16像素”抽查256像素的全贡献CPU参考，未漏掉最强在内分量；最大投影差 {max(p['max_pixel_error'] for p in exact['projection_precision']):.8f}像素。最差局部 K64 质量仅 {min(p['retained_mass'] for p in exact['pixels']):.5f}，不能把相机平均当作逐像素保证。截断损失最大绝对差 B1={methoderrors['B1']:.6g}参考像素、B2={methoderrors['B2']:.6g}、H={methoderrors['H']:.6g}、M={methoderrors['M']:.6g} NLL；H先验归一化也受截断影响。K64只是当前短序列经过核验的共同设置，长序列仍需监测关键候选与截断。

## 实测资源（单RTX4090）

时间包含三帧数据设置、缓存读取、训练诊断及保存，排除可复用frame1初始化和教师预计算。总进程wall time、失败和初始化索引见command ledger；不能称为纯kernel训练时间。

| 场景 | 拓扑 | 方法 | 三帧训练与保存秒 | 峰值allocated/reserved GiB | 末帧点数 | 固定λ |
|---|---|---|---:|---:|---:|---:|
{table}

共20单元训练与保存累计 {sum(r['train_and_save_seconds'] for r in rows)/60:.2f}分钟。最大allocated {max(r['peak_allocated_gib'] for r in rows):.2f}GiB。当前Python分块实现对全像素K64有明显耗时成本；不声称高效率。正式分辨率的共享头资源尚未实测。

## 独立验证视角烟雾评估

仅 cam01，三个更新帧，复用阶段03统一FP32 PSNR/SSIM/LPIPS定义，原生完整状态与decoded snapshot渲染回归通过。以下数字只检查管线和明显异常，不能用三帧小差异选择最优方法或宣称改善；没有读取cam00调参。

| 场景 | 拓扑 | 方法 | PSNR | SSIM | LPIPS |
|---|---|---|---:|---:|---:|
{quality}

## 恢复、版本和失败记录

M原生coffee从frame2完整checkpoint分别续训到frame4两次，固定λ完全继承；UID唯一、实际offset优化器连接正确，点数 {resume['points']}。渲染续训差通过阶段03同样的重复波动容差与平均1/255绝对上限；不声称CUDA逐位确定。详细见 `manifests/phase04_resume_check.json`。

根目录已建立本地Git，`phase03-validated`为原基线代码/配置/报告索引快照；每个实现版本单独提交。代码、配置、必要补丁、测试、报告和小产物索引在Git；数据、环境、教师缓存及checkpoint保存在磁盘并由哈希索引追溯。新运行记录Git commit和入口模块SHA；早期运行保留其版本上下文。早期coffee B1的投影分母版本已移到 `runs/phase04_versions/v2/frozen_coffee_martini_B1`，并使用当前共同投影重跑，最终表不使用旧产物。

command ledger保留 {len(errors)} 个失败执行：首次脚本 import路径错误、源native预计算颜色的空mask指针访问错误、设备记录字段重名错误；均修复后重测。准备阶段系统Python缺PyYAML及评估生成器空格匹配失败也发生过，已分别使用既有隔离环境Python和修正生成器，未伪造其成功。原B0的4,140个非PT锁定文件完整哈希一致；480个原快照确认存在，但此轮未重哈希其大文件内容，边界明确。

## 与B0分离的共同协议修正

B0使用当前帧网格与原线性化向量投影流；共享头使用源网格、完整端点投影、冻结贡献、持久UID、固定教师FB/边界有效mask、规范化参考像素单位和梯度尺度校准，另有共同offset修复。共享缓存使用已冻结FP16和轴向resize修正，B0继续其FP32缓存与原mask规则。上述差异全部属于共享适配层，不能算作M相对B2/H的独立贡献。阶段05应分别回答共享适配收益与混合归属收益。

## 已执行入口与下一阶段

```powershell
python rank1_experiments/scripts/phase04_launch.py tests rank1_experiments/scripts/phase04_tests.py
python rank1_experiments/scripts/phase04_launch.py real_probe rank1_experiments/scripts/phase04_real_probe.py
python rank1_experiments/scripts/phase04_launch.py exact_pixels rank1_experiments/scripts/phase04_exact_pixels.py
& 'rank1_experiments/envs/morgs-cu121/python.exe' rank1_experiments/scripts/phase04_short_batch.py
& 'rank1_experiments/envs/morgs-cu121/python.exe' rank1_experiments/scripts/phase04_resume_jobs.py
& 'rank1_experiments/envs/morgs-cu121/python.exe' rank1_experiments/scripts/phase04_eval_batch.py
```

这些入口实际执行过；详细子命令、路径、返回码、耗时和日志SHA在 `manifests/phase04_commands.jsonl`。short_batch跳过已经完成的单元，不覆盖原B0。产物索引 `phase04_checkpoint_index.json`、完整表 `phase04_short_table.csv`；图在 `diagnostics/phase04/`。

下一阶段为05：两个开发场景90帧、有限且同预算的λ/σ搜索，固定验证规则，比较M对B2和H，持续报告原密度的UID和K覆盖。当前只确认实现与短训可用；阶段05未启动。
'''
    (ROOT/'reports/phase04_implementation.md').write_text(report,encoding='utf-8')
    (ROOT/'phase04/README.md').write_text('# Phase04\n\nImplementation gate PASS. See ../reports/phase04_implementation.md.\n\nCommon K64, sigma2, epsilon0.02. The original B0 renderer and phase03 artifacts are frozen.\nShort runs validate implementation only. Phase05 has not run.\n',encoding='utf-8')
    paths=[]
    for pattern in ('scripts/phase04*','phase04/*','configs/phase04/*','patches/phase04*','manifests/phase04*','diagnostics/phase04/*'):
        paths.extend(p for p in ROOT.glob(pattern) if p.is_file() and p.suffix!='.lock')
    paths += [ROOT/'reports/phase04_implementation.md',ROOT/'working/MoRGS/train_phase04.py',ROOT/'scripts/run_phase04.cmd']
    for row in rows:
        run=ROOT/row['run']
        for pattern in ('*.json','*.jsonl','*.txt','*.csv','identities/*.pt','snapshots/*.pt','evaluation/validation/*.json'):
            paths.extend(run.glob(pattern))
    for directory in (ROOT/'runs/phase04/resume',ROOT/'runs/phase04_versions'):
        paths.extend(p for p in directory.glob('**/*') if p.is_file() and
                     (p.suffix in ('.json','.jsonl','.csv','.txt') or
                      (p.suffix=='.pt' and p.parent.name in ('snapshots','identities'))))
    for p in paths:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
    lock={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(paths)
          if p.name!='phase04_artifact_lock.json'}
    lockpath=ROOT/'manifests/phase04_artifact_lock.json'
    lockpath.write_text(json.dumps(lock,indent=2),encoding='utf-8')
    statepath=ROOT/'experiment_state.json';state=json.loads(statepath.read_text(encoding='utf-8'))
    state.update(phase='04',status='PASS',phase04='PASS',continue_gate_passed=True,next_phase='05',
        updated_at=datetime.now().astimezone().isoformat(),phase04_complete=True,phase05='NOT_STARTED',
        active_artifact_lock='manifests/phase04_artifact_lock.json',
        patch_sha=hashlib.sha256((ROOT/'patches/phase04_train_adapter.patch').read_bytes()).hexdigest(),
        phase04_report='reports/phase04_implementation.md',phase04_common_k=64,
        phase04_summary=dict(short_runs=20,updated_frames_per_run=3,methods=['F0','B1','B2','H','M'],
                            uid_min=min(r['uid_mass_min'] for r in flow),topk_min=min(r['topk_mass_min'] for r in flow),
                            max_allocated_gib=max(r['peak_allocated_gib'] for r in rows),quality_improvement_claimed=False),
        scope='Phase04 implementation and short-training gate passed; phase05 effectiveness study not started')
    state['manifests']['phase04_artifact_lock.json']=hashlib.sha256(lockpath.read_bytes()).hexdigest()
    statepath.write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(status='PASS',locked_files=len(lock),report=str(ROOT/'reports/phase04_implementation.md'))))


if __name__=='__main__':main()
