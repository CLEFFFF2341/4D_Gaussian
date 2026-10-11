# 阶段05运行说明

数值候选头直接复用已锁定的phase04模块，不改原RGB后端。`phase05_protocol.json`在读取本轮验证成绩前冻结：两个场景、seed0、90帧、676×507、原密度、K64、σ2；B1/B2/H/M各三个ratio，全部24单元。同一frame1初始化、相机采样状态与共同训练checkpoint校准，λ在每个单元中固定。

cam01代理区域由独立SEA-RAFT推理冻结，评价进程独占这些文件。训练不能读validation/test RGB。每次记录完整配置、命令、入口哈希、资源、逐帧相机采样SHA、候选覆盖、posterior与实际运动梯度。

## 已执行启动检查

`phase05_tests.py`核验区域SSIM与原指标一致、FB边界、空区域、选参和平局规则、跨场景/LPIPS门槛；`phase05_startup_check.py`实际校准四个家族，运行真实frame2训练、完整checkpoint与snapshot回归、区域评估。前期语法/导入/CSV合并错误修复提交均保留，不删除失败日志。检查通过证据在`manifests/phase05_tests.json`与`phase05_startup_check.json`。

## 串行队列与状态

入口：`scripts/phase05_runner.py`，用`envs/morgs-cu121/python.exe`运行。它冻结180个验证帧mask、校准、独立重评B0、跑两个F0诊断，然后按0.1→0.03→0.3依次运行所有方法与场景；完整单元随后评估与提交Git小产物索引。每20秒更新`phase05_progress.json`，不占用另一个GPU任务。

查看状态：

```powershell
python rank1_experiments/scripts/phase05_status.py
```

最终成绩在`reports/phase05_go_no_go.md`；实际日志见`manifests/phase05_started_commands.jsonl`与`phase05_commands.jsonl`。进程退出或覆盖门槛失败时记录failure，保留partial run。完成checkpoint边界的资源和诊断一致时可恢复；中断后先核查记录PID，不能同时启动两套队列或盲目删除lock。已完成trial不覆盖，恢复前核查固定代码和协议哈希。

## 搜索后边界

单种子无信号时只读诊断一次，停止扩规模；有信号时状态为`REPEAT_REQUIRED`，不能称阶段05通过。正式cam01树、共享头正式分辨率预算及seed1/2初始化当前尚未齐备，需继续阶段05的300帧两场景三种子复核。阶段06不会由这个队列启动。此队列完成的是预注册seed0初筛部分，不是把所有未来复核冒充已执行。

本地Git提交保存每个代码版本和每个完整trial索引，不推送。数据/环境/模型大文件留在磁盘，checkpoint完整SHA记录在trial索引；原阶段03/04锁定产物不覆盖。
