# 阶段 03：B0 基线与实际预算

更新时间：2026-10-11T03:21:37.977299+08:00。状态：执行门槛通过。

本阶段保留固定 MoRGS 的损失、增点和逐帧更新，初始10000 iterations；后续8 epochs/view。缓存适配、Windows必要修复与独立评估使本地运行应标为 **B0 compatibility / 严格数据协议**，不能直接称为作者论文数字复现。

开发两场景使用阶段02训练视角稀疏初始化；正式 coffee_martini 使用 COLMAP 稠密初始化（如果已生成），所有初始化仅使用本划分训练视角第0帧。不同初始化的指标不用于判定算法增益。

| 运行 | 已保存帧 | 统一PSNR / SSIM / LPIPS | 训练+保存秒 | allocated / reserved GiB |
|---|---:|---|---:|---|
| dev coffee_martini | 90/90 | 20.79437/0.82015/0.27967 | 391.7 | 1.196 / 1.531 |
| dev cut_roasted_beef | 90/90 | 21.92455/0.85008/0.25951 | 443.8 | 0.972 / 1.229 |
| final coffee_martini | 300/300 | 28.75449/0.91589/0.20831 | 2408.8 | 3.089 / 4.576 |

渲染FPS含一次warmup、每个保留checkpoint测量5次原GaussianModel（包括latent decoder）同步渲染，加载不计入；另存decoded snapshot FPS，二者不混称。
- dev coffee_martini: full model 355.60 FPS; snapshot 575.73 FPS; evaluation 13.1s.
- dev cut_roasted_beef: full model 425.32 FPS; snapshot 613.79 FPS; evaluation 11.4s.
- final coffee_martini: full model 234.18 FPS; snapshot 290.92 FPS; evaluation 104.0s.

## 成本与复用

| 运行 | 首帧10000迭代+保存秒 | 后续训练帧计时秒（含保存） | CPU帧边界RSS最大GiB | 当前磁盘GiB |
|---|---:|---:|---:|---:|
| dev coffee_martini | 94.2 | 271.2 | 1.656 | 8.066 |
| dev cut_roasted_beef | 128.2 | 285.5 | 1.602 | 5.211 |
| final coffee_martini | 270.7 | 1933.1 | 2.111 | 37.763 |

首帧checkpoint是该主干seed0的分叉起点。每个方法独立从零的成本需加上首帧训练；共享初始化时每个额外分叉可节省表中首帧时间。coffee开发主运行已从同协议smoke首帧复用；其成本表仍计入首次初始化成本。每帧资源记录保存checkpoint大小，最终索引保存全部可恢复文件SHA256。

教师和稠密初始化为可复用的预处理成本；精确秒数和显存见 resource_budget.json 的 teacher_dev、teacher_final、dense_initialization 字段。实际命令的进程墙钟时间含启动、加载与I/O，见 manifests/phase03_commands.jsonl，父流水线时间不可与其子作业重复求和。

已有开发可视化显示场景整体对齐，但留出视角背景和边缘存在覆盖/纹理误差。稀疏与稠密初始化结果不能归为监督方法收益；其原因未由单变量对照完全隔离。此阶段的PASS仅代表所列执行检查，论文复现和方法有效性仍未成立。

正式cam00第299帧的桌面、窗户、柜子和人物整体对齐，仍有细纹理和手部/躯干误差。正式300帧指标曲线稳定；开发cut后段PSNR下降，尚未单变量隔离内容变化、初始化与原运动更新的影响。可視化核验见 phase03_visual_qa.json。

## 可审阅产物

- 完整恢复索引：checkpoint_index.json；实际命令：manifests/phase03_commands.jsonl。
- 逐帧数值：各run的evaluation/validation或evaluation/test/per_frame.json及resources.jsonl。
- 图像与曲线：figures/phase03/final_coffee_martini_heldout.png、quality_curves.png、resource_curves.png。
- 冻结文件SHA256：manifests/phase03_artifact_lock.json。

## 评估与隔离

训练进程不构造测试 loader，并通过图像文件打开守卫拒绝 validation/test PNG。开发验证 cam01 与正式最终测试 cam00 只在单独评估进程读取，测试数字不用于选超参数。原训练日志中 Val 字段实际是训练cam0的诊断，不能当开发验证成绩；真正成绩只看 evaluation/validation。

统一 PSNR 用FP32 RGB MSE，SSIM沿用11×11 sigma1.5的实现，统一LPIPS以VGG0.1、RGB*2−1输入。作者命名空间另用8位量化PSNR/SSIM和原[0,1] LPIPS。相同图像、已知常量误差与LPIPS对称性自检见 eval_protocol.json / phase03_eval_selftest.json。

所有本次运行 use_xyz_legacy=False，遵循官方CLI默认的_xyz_fixed路径。官方参数帮助另将--use_xyz_legacy标为论文数字复现入口，因此本次不满足该论文配置复现声明。此选择先于正式测试冻结，未据测试成绩切换。

每个run内cfg_args保存执行时参数。开发运行的教师路径字段仍为原相对路径，但缓存适配器实际加载ROOT/weights固定SHA的同一教师；最终YAML将这些字段绝对化，研究超参数不变。恢复和评估以实际状态与缓存manifest为准。

## 保存、恢复与源码边界

checkpoint 是帧边界完整状态：Gaussian属性/解码器/offset/previous attributes/mapping/optimizer/相机缓存、采样随机状态、Python/NumPy/CPU/CUDA RNG和split generator。每帧更新latest；保留首帧、2/5/10帧、每30帧与最后一帧。逐帧FP32解码快照用于独立渲染，不能代替恢复checkpoint。

10帧连续运行与从第5帧中断后的续跑及同checkpoint重跑已比较（状态以manifest为准）。CUDA执行下重跑也有增点和RGB差异；恢复误差必须不高于两次重跑差异的2倍，并限制平均RGB差异不超过1/255。局部最大误差仍可较大，这不是逐位确定性保证。其来源尚未隔离，不把所有差异直接归因于CUDA atomics。

发现作者 prune_points 将 offset_3d 切片，未绑定到 _prune_optimizer 返回的替代参数；因此增点后offset可能与optimizer组脱离。B0参考保留该行为并完整保存两个对象，任何修复必须另列共享控制，不能算混合监督贡献。

教师构造的随机状态消耗及TF32设置在训练前重现，随后卸载；训练只读固定FP32 B0缓存。共享FP16 resize/FB-valid协议未用于B0。

阶段03核对发现：阶段02缓存的图像归一化在CUDA完成，而真实ToTensor loader在CPU完成，舍入微差可产生约0.0065px全缓存最大流差及少量SAM union像素变化。已新建 dev_phase03 缓存并冷启动重训，旧cache/teachers/dev和旧runs/.../seed0仅为工程诊断。训练入口抽查在线教师/缓存flow、heatmap、union与loss相同，xyz梯度最大差约6.5e-9。证据见 phase03_cache_input_precision.json / phase03_teacher_training_equivalence.json。

## 初始化核验

COLMAP工具来自官方3.8 Windows CUDA发行包，SHA256 60910e2d79539ba8825628bd147f900a5cd61646b5a9ad5f62dbec0e9b78e8d0。参考作者链接的 multipleviewprogress.sh 完成稠密深度融合与至多40000点的voxel下采样。为保持发布坐标和测试隔离，使用公开固定标定+point_triangulator，而非重跑mapper并重新估计LLFF位姿；NumPy替代Open3D voxel实现。这些差异明确保留，尚未声称点云与作者相同。

实际初始化命令和读入图像SHA存于 phase03_dense_final_coffee_martini.json。原稀疏点云与旧研究输出保留。

## 继续条件

只有两开发场景90帧、一个正式300帧均能训练与独立评估，恢复检查通过并有显存/耗时实测，才允许进入阶段04。只打通训练或只保存部分帧，状态保持RUNNING。论文质量复现另需逐项核对初始化、作者配置/legacy模式及参考论文指标，不能用本报告自动替代。
