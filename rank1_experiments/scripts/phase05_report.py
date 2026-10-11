"""Incremental report; unavailable trials never become invented results."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase05.common import read,write,now,state_update
from phase05.selection import select,screen

def main():
    paths=list((ROOT/'manifests/phase05_trials').glob('*.json'))
    rows=[read(p) for p in paths];search=[r for r in rows if r['method']!='F0']
    progress=read(ROOT/'manifests/phase05_progress.json') if (ROOT/'manifests/phase05_progress.json').exists() else {}
    complete=len(search)==24
    winners=select(search) if complete else None
    gate=screen(search,winners) if complete else None
    if gate:write(ROOT/'manifests/phase05_screen.json',dict(status='SEED0_SCREEN_COMPLETE',winners=winners,**gate))
    display=lambda v:'—' if v is None else f'{v:.4f}'
    table='\n'.join(f"| {r['scene']} | {r['method']} | {r['ratio']:.2f} | {r['metrics']['PSNR']:.4f} | {r['metrics']['SSIM']:.4f} | {r['metrics']['LPIPS']:.4f} | {display(r['regions']['boundary']['PSNR'])} | {r['seconds']:.1f} | {r['peak_allocated_gib']:.2f} |" for r in sorted(rows,key=lambda r:(r['scene'],r['method'],r['ratio'])))
    body=f'''# 阶段05：最小混合方法 go/no-go

更新：{now()}。**当前状态：{progress.get('status','PREPARED')}，完整90帧搜索已完成 {len(search)}/24 单元，F0诊断 {sum(r['method']=='F0' for r in rows)}/2。**

本报告随实际单元完成更新。未完成全部24组前不选参，不给M有效性结论。阶段04的短训不能代替阶段05成绩；阶段06尚未启动。

## 已冻结协议

coffee_martini、cut_roasted_beef，seed0，前90帧，676×507，K64，σ2，ε=.02，原密度控制、八轮/视角与共同初始化。B1/B2/H/M均只搜索梯度比例0.03/0.1/0.3共三组，执行顺序0.1→0.03→0.3。所有家族预算相同；不追加只针对M的σ搜索。每组完整运行90帧，不按中间质量提前停止。每场景所有方法从相同B0 frame1 checkpoint恢复，逐帧保存实际采样序列SHA并比较。

在同一RGB-only训练前缀的首次flow反传之前，同时计算四种未加权损失与RGB对共享运动参数的梯度范数，以此冻结每场景/家族unit λ。每个run的λ=ratio×unit λ，不再动态调整。开发时允许按场景校准；复核若进入则使用已冻结跨场景全局λ，不按种子重搜。原phase04数值头不改；phase05只加记录与调度。

验证仅cam01；使用冻结SEA-RAFT独立推理生成评价代理区域，训练进程拒绝读取validation/test RGB，也不载入代理区域。动态=固定FB有效且参考像素位移>1.5；边界=该二值区域的形态梯度，开发半径3像素（参考6）；静态=FB有效非动态；teacher_inconsistent=固定FB无效，是噪声/遮挡代理，不能称教师错误真值。末帧采用自身网格上的反向流。区域SSIM为完整图SSIM map上的masked mean，区域PSNR为每帧区域RGB MSE的dB再平均。空区域记missing及像素数。

指标沿用阶段03 FP32/11×11 SSIM/VGG LPIPS定义。家族统一按两场景等权整体PSNR选一个ratio，1e-6以内平局依次看LPIPS和较小ratio；不能为每个区域另选赢家。初筛至少一场景较其最强选定B1/B2/H提高0.05dB，另一场景退化不超过0.05dB；或两场景边界均提高0.10dB且整体退化不超过0.05dB；每场景LPIPS退化不超过.002。此为单种子信号，不能代替三种子门槛。

B0复用原90帧作者兼容结果，新增区域评估写到runs/pilot/references，绝不写原B0目录。F0每场景完整90帧，仅辅助诊断。B0与共享头之间的source grid/真实投影、FP16与轴向resize修正、有效mask、身份、offset修复等区别见阶段04报告；M必须独立超过共享B2/H才能支持混合归属的贡献。

## 全部已完成试验

空区域显示为缺失；原JSON保留null。耗时为更新frame2—90训练、诊断及保存，初始化和教师预计算分别记录，不能称纯训练kernel耗时。

| 场景 | 方法 | 比例 | PSNR | SSIM | LPIPS | 边界代理PSNR | 更新与保存秒 | 峰值allocated GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{table or '| 尚无完整90帧单元 | — | — | — | — | — | — | — | — |'}

每帧指标在各run的metrics_per_frame.csv及evaluation/validation/per_frame.json；流/RGB范数、候选与异常后验分布、motion confidence梯度、UID/K/像素覆盖、点数及资源保留在各run和trial索引。固定协议、命令与失败见manifests/phase05_protocol.json、phase05_started_commands.jsonl、phase05_commands.jsonl。执行状态见phase05_progress.json。

## 当前研究判断

'''
    if not complete:
        body+='M是否超过B2/H、是否超过强控制、收益区域、原密度下是否成立及额外资源，均等待完整搜索，当前不作结论。\n'
    else:
        lines=[]
        for d in gate['differences']:
            lines.append(f"- {d['scene']}：M−B2={d['vs_B2']:+.4f}dB，M−H={d['vs_H']:+.4f}dB；相对最强{d['strongest_control']}整体{d['whole']:+.4f}dB，LPIPS{d['lpips']:+.5f}，边界代理差{d['boundary']}dB。")
        body+='\n'.join(lines)+'\n\n'
        body+=('初筛出现信号，**仍未通过阶段05重复验证门槛**。下一步为M与两个最强控制的两场景300帧seeds0/1/2复核，需要正式cam01验证数据与共同分辨率资源核验；不得自动进入阶段06。\n' if gate['signal'] else '预注册初筛无信号。执行一次只读故障诊断，不追加选择性搜索、gate或轨迹模块；未发现可修复的共同错误时停止扩规模，保留负结果。\n')
        selected={m:dict(**w,global_lambda=sum(next(r['lambda_flow'] for r in search if r['scene']==s and r['method']==m and r['ratio']==w['ratio']) for s in ('coffee_martini','cut_roasted_beef'))/2,
                        freeze_policy='arithmetic mean of two training-checkpoint calibrated lambdas; no seed-specific retuning') for m,w in winners.items()}
        write(ROOT/'configs/frozen_pilot_configs/selected.json',dict(status='FROZEN_FOR_PENDING_REPEAT',methods=selected,validation_selection='two-scene macro PSNR'))
        controls=sorted(('B1','B2','H'),key=lambda m:winners[m]['PSNR'],reverse=True)[:2]
        write(ROOT/'manifests/phase05_repeat_requirements.json',dict(status='REQUIRED' if gate['signal'] else 'NOT_TRIGGERED',methods=['M',*controls],scenes=['coffee_martini','cut_roasted_beef'],frames=300,seeds=[0,1,2],
              current_unmet=['Formal cam01 validation tree is not yet extracted','Formal resolution shared K64 head feasibility/time/memory has not been measured','Seed1/2 common initialization checkpoints not yet generated'],next_phase='05'))
    if search:
        seconds=sum(r['seconds'] for r in search)/sum(r['updated_frames'] for r in search)
        disk=sum(r['disk_bytes'] for r in search)/sum(r['updated_frames'] for r in search)
        body+=f'''\n## 后续成本估计\n\n当前每个更新帧含诊断与保存平均{seconds:.2f}秒。阶段06若按两场景四变体90帧，原型预算约{seconds*89*8/3600:.2f} GPU小时，目标可见性额外成本尚未实测，先给1—2倍范围。阶段08若六场景、三种子、三方法、300帧，相同开发分辨率外推约{seconds*299*54/3600:.1f} GPU小时与{disk*299*54/1024**3:.0f}GiB存储；1352×1014四倍像素的共享头尚未核验，暂用2—6倍耗时范围。这是成本外推，不是正式测量；教师、初始化、第二数据集和原B0另计。\n'''
    else:body+='\n## 后续成本估计\n\n阶段04原型显示本轮可能需要14—22 GPU小时及160—300GiB；在首个90帧单元完成后更新阶段06/08估计。正式共享头尚未实测。\n'
    body+='\n原稀疏开发初始化的背景覆盖限制仍存在，单cam01与两场景不能支持广泛结论。相邻帧不被当作独立统计样本。\n'
    path=ROOT/'reports/phase05_go_no_go.md';path.write_text(body,encoding='utf-8')
    print(json.dumps(dict(status=progress.get('status','PREPARED'),complete_search=len(search),signal=None if not gate else gate['signal'])))

if __name__=='__main__':main()
