# 阶段 02：数据、划分与教师缓存审计

更新时间：2026-10-11T01:12:02.305536+08:00。状态：**通过开发数据门槛**。完整 MoRGS 论文复现尚未完成，未运行场景训练。

## 实际完成范围

复用 E:\4DGS\data 中的原始发布视频，不修改旧实验。coffee_martini 18 路、cut_roasted_beef 20 路均完整提取前 300 帧（1352×1014），开发片段为 0—89（676×507）。共 14,820 张 PNG；每张已完整解码并保存 SHA256，最终再次核对全部文件哈希。六场景 120 路视频均核对全部容器 PTS，0—299 使用一致的 30 FPS 时间网格。其余四场景已下载的原视频列入清单，尚未做完整顺序 RGB 解码和图像导出。

同步证据来自容器 PTS 与 [N3DV 官方发布说明](https://github.com/facebookresearch/Neural_3D_Video)，不是独立物理同步测量。公开发布没有可复核的完整视频官方 SHA；本地哈希用于锁定字节，前阶段已对两个开发场景的 ZIP 作 CRC 检查。

## 冻结划分与读入隔离

cam00 为最终测试；开发验证为 cam01，其余真实可用视角训练。正式训练恢复 cam01。poses_bounds.npy 的原始行对应实际 MP4 排序，不能用相机编号直接当行号。派生目录中的 poses_bounds.npy 是该角色相机按原顺序截取的子集。公开全相机标定允许使用；初始化与教师只读取训练 RGB，第 0 帧点云也不使用未来 RGB。训练目录没有测试或开发验证图像。

| 场景 | 开发训练视角 | 正式训练视角 | 光流原相机 | 开发运行时索引 |
|---|---:|---:|---|---|
| coffee_martini | 16 | 17 | cam02, cam06, cam10, cam16 | [0, 3, 7, 12] |
| cut_roasted_beef | 18 | 19 | cam02, cam06, cam10, cam15 | [0, 3, 7, 12] |

实际 loader 与 scene reader 已用图像文件打开追踪验证四个训练目录只读其训练 RGB；错误使用默认 test_indices=[0] 或把测试根目录传给训练 loader 会明确拒绝。该补丁只处理读入，不修改研究损失。**原 train.py 仍把测试数据和训练循环绑定，阶段 03 必须接入独立验证/最终评估与缓存 adapter；不能直接把旧命令当作严格隔离训练入口。**

## 相机、缩放和初始化

LLFF down/right/back 转成 CV right/down/front，w2c=Rᵀ、t=−RᵀC。原图 2704×2028，无裁剪；FFmpeg area 缩小 2 倍得到正式图像，再由 OpenCV INTER_AREA 缩小 2 倍得到开发图像。像素中心用 (x+0.5)s−0.5；流的 x/y 分量分别乘对应尺寸比例。保持 B0 的对称 FoV 光心 (W−1)/2；它和严格原 K 的 half-pixel 缩放在开发尺寸相差 0.125 像素、正式相差 0.25 像素，分别记录，未悄悄改作者投影。poses 不含畸变系数，沿用发布图像处理，不声称重新标定或去畸变。

**初始化协议差异：** 新点云由公开已标定相机和训练第 0 帧的 SIFT 对应三角化生成。它是可审计的稀疏开发初始化，尚不是作者链接的 COLMAP dense / downsample2 点云流程；文件同名不代表等价复现。开发和正式划分分别重建，不复用漏掉 cam01 的点云冒充正式初始化。阶段 03 必须将这套结果标为严格开发协议；与论文数字比较前补做作者初始化流程，所有方法共同使用同一初始化。

| 场景/划分 | 点数 | 正确位姿极线内点比例中位数 | 错配位姿对照 |
|---|---:|---:|---:|
| coffee_martini/dev | 7167 | 76.774% | 0.893% |
| cut_roasted_beef/dev | 9232 | 79.024% | 1.149% |
| coffee_martini/final | 7570 | 75.781% | 1.149% |
| cut_roasted_beef/final | 9604 | 79.279% | 0.766% |

重投影使用真实训练图像对应，并检查正深度、三角化角度与各视角误差。接受点的各视角重投影误差小于 1.5 开发像素；极线对应独立于三角化筛选统计。错配位姿对照显著更差，支持实际排序和图像方向正确；这不是 3D 运动真值验证。

## 教师与缓存

SEA-RAFT M 与 SAM2.1 large 使用阶段 01 严格加载的固定权重，SHA/配置/源代码/相机/帧号/单位写入 manifest 和缓存。对两场景各 4 个真实训练视角、90 帧预计算：1,440 份方向流（89 个相邻对和额外 0→3 初始化对的前后向），88 份 SAM 刷新标签。正式 300 帧的教师缓存尚未生成，将按同一固定协议在阶段 03/正式运行前补齐。

原版 B0 保留 FP32 未过滤光流与 heatmap、1.5 像素运动阈值和原版 SAM 跨相机列表顺序；第一个训练 step 用 0→3，之后用前帧→当前帧；SAM 在 step 1、2、10…90 刷新并保持到下一次刷新。未改 [0,1] 输入，也未修正上游注释掉的 RAFT normalization 或 SAM 自定义预处理。B0 缓存读回使用全像素有效掩码，绝不混入共享 FB 掩码。

共享候选独立保存 FP16 流：修正上游 507→253→506→507 最后 resize 漏掉的 y 分量倍率（507/506）；输入教师不升级。共享 FB mask 由未量化 FP32 计算，在参考 1352×1014 像素单位使用 ||f+warp(b)||² ≤ 0.01(||f||²+||warp(b)||²)+0.5，并与越界和有限数值条件相交。此修正是共同工程对照，不能计入新损失的研究收益。

在线流抽查 24 对，B0 缓存往返最大误差 0；在线重复最大差 0 像素。SAM 两场景各 4 视角重复均无变化像素。FP16 最大绝对误差 0.0019516945，全缓存平均 EPE 误差 4.8560504e-06 开发像素。共享 mask 覆盖 min/mean/max 为 96.990%/99.576%/99.914%。B0 全部 720 个相机×训练 step 重放通过；这是标签/调度等价验证，训练过程端到端等价仍须阶段 03 接入后检查。B0 SAM 联合运动 mask 平均覆盖仅 0.206%，已记录而不调高覆盖；阶段 03 检查其对运动信心激活的影响，不能将数值一致性当教师运动准确性证明。

教师计算顺序为 RAFT 后卸载、SAM 后卸载。RAFT 进程峰值 allocated 0.185 GiB，SAM 1.597 GiB；总阶段 RAFT 壁钟 1530.1 秒包含等待图像导出，不能当纯推理时间或训练预算。SAM 壁钟 16.6 秒。缓存重放只读取文件，没有教师模型；阶段 03 不应常驻这两个模型。

## 导出错误与处理

首次 FFmpeg 直接编码 PNG 返回成功，但 coffee/cam00 发现 3 张 IDAT 无法解码（formal 0043/0217、dev 0047）。已改为严格视频解码→原始像素管道→OpenCV PNG 编码，并逐张 Pillow 完整解码。cut/cam00 初次在 frame 190、cut/cam19 在 frame 122 被 -xerror 中止，保留日志后分别重新完整导出并通过。任何失败尝试都不计为成功，不跳帧，不修改原视频。最终通过帧数、完整解码、哈希的全部结果才进入 manifest。错误根因未独立确定，不能声称原文件有不可恢复损坏或机器故障。额外重试记录见 extraction_failures.json 和 logs/*failed*。

## Meet Room

[StreamRF 官方入口](https://github.com/AlgoHunt/StreamRF#dataset) 的 [Drive](https://drive.google.com/drive/folders/1lNmQ6_ykyKjT6UKy-SnqWoSlI5yjh3l_) 有 discussion、trimming、vrheadset，每场景 13 路 cam_0…cam_12，位姿均为 13×17 有限数组，公开标定尺寸 1280×720。官方 prepare_dataset.py 将 cam_0 改名 cam00；官方 meetroom.json 的 llffhold=100 与 loader 的 ind % hold_every 对应首视角留出；MoRGS 默认 test_indices=[0]，论文指中央视角，位姿中 row0 也都最接近相机中心。因此冻结原始测试视角 cam_0，不能盲套 N3DV 名称。使用数值后缀排序与零补齐映射，避免 cam_10 字典序错位。镜像额外 stepin 不纳入三场景协议。ModelScope tree 可达，但文件下载要求登录；通过无需登录的官方 Drive 取得小型位姿与列表，未下载 39 路视频。图像级重投影、视频帧数/FPS/完整性仍待第二数据集实际准备。

## 产物与执行

关键清单：manifests/dataset_manifest.json、splits_dev.json、splits_final.json、teacher_manifest.json、teacher_cache_verification.json、phase02_loader_checks.json、video_timestamp_audit.json、meetroom_manifest.json。每帧索引在 manifests/extraction/<scene>/<camera>.json。入口均在 scripts/，以绝对脚本路径调用 run_official.cmd。

```powershell

& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase02_prepare_data.py' extract
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase02_geometry.py' --protocol dev
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase02_geometry.py' --protocol final
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase02_teacher_cache.py' raft
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase02_teacher_cache.py' sam
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase02_protocol_checks.py' loader
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase02_cache_verify.py' 
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase02_finalize.py' 
```

这些命令已经执行；extract/教师支持已完成单元的续跑，geometry 会重新构建对应点云。不要再次运行 prepare_data.py init 覆盖冻结划分。阶段 01 finalizer/handoff 属于历史阶段，也不要运行来覆盖阶段 02 状态。

## 继续门槛与下一步

两开发场景全部视角/帧完整、训练图像和位姿对应、缓存数值与调度抽查、开发/测试读入隔离均通过。允许进入阶段 03 的严格开发基线：先完成独立评估与缓存接入、可恢复状态，再执行 10 帧 smoke。90 帧/正式 300 帧训练尚未开始，PSNR、训练耗时与峰值显存没有结果，不能宣称方法有效或完成论文复现。
