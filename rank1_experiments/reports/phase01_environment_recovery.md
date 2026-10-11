# 阶段 01 续跑：环境与教师障碍已解除

**阶段 01 的环境／源码继续门槛通过。** 官方核心版本环境、定制算子、预训练教师、模型级反向更新及评估入口均已执行。阶段 02、03 尚未开始；没有完整场景训练、基线质量、FPS 或场景显存／耗时结果。当前状态以 `experiment_state.json` 为准，首轮 PARTIAL 保留为历史记录。

## 隔离环境恢复

环境 `E:\4D_Gaussian\rank1_experiments\envs\morgs-cu121`：Python 3.10.18、PyTorch 2.5.1、torchvision 0.20.1、CUDA runtime 12.1，隔离 nvcc 12.1.105、VS2019、sm_89。没有复用旧环境 site-packages，未改系统 toolkit、驱动、代理／SSL 全局设置，未重启或重置 GPU。

上轮大部分依赖已缓存；离线创建只缺 `libcusparse-dev-12.0.2.55-0.tar.bz2`。恢复后 SHA256 与冻结元数据一致：`2bda05db5ced35dd2877bbb55e90ea186ed2886d8cfdbc3d1f7542278211b255`。补归档后 conda 仍尝试远程查找，故把原始 URL 逐项映射为本地归档，用 local explicit 清单离线创建成功。

远程 URL／每包哈希、实际 conda 包及 pip freeze 均保存。作者未钉死的依赖采用明确版本，核心 torch/CUDA 未升级，pip check 无冲突。包列表的 cuda-version 元包不等于编译器版本；以后不能重新求解后静默换 toolkit。

Windows 代理及 TLS 吊销查询表现不一致，下载选择实测可用的官方端点连接。curl 的 `--ssl-revoke-best-effort` 容许离线吊销查询失败，仍校验证书；没有全局关闭 SSL 校验。

## 算子与模型检查

| 检查 | 实际结果 |
|---|---|
| CUDA 张量 | 目标 torch 前向、反向、同步通过 |
| simple-knn | 三近邻平方距离均值与 torch 参考最大差 1.86e-7 |
| torch_cluster | pt25/cu121 Windows wheel；32 点、k=3，96 条无自环边 |
| gradient rasterizer | SH/RGB/flow 前后向及八列掩码回归通过 |
| ordinary rasterizer | 非空渲染、有限参数梯度、xyz 梯度非零 |
| SAM2 CUDA | Windows pyd 编译通过；连通域面积 4 和 6 正确 |
| torchac | MSVC 编译后 12 个符号编码／解码逐元素一致 |
| 实际 GaussianModel | 32 点、首帧／第二帧各一步 render→loss→backward→Adam；两帧 xyz 梯度与真实坐标更新非零，第二帧 optimizer 含 offset |

普通 rasterizer 最初的 z=3 测试点被它的硬编码 `p_view.z<=4.0f` 裁剪；换 z=6 后通过，只调整测试输入。主 rasterizer 保留前轮掩码修正，没有改光流尺度。

SAM2 的 Linux .so 被 Windows CUDA 扩展补足；来源为 [官方固定文件](https://github.com/facebookresearch/sam2/blob/f882beb157546530a7e66a5f72ef1da4f74593d7/sam2/csrc/connected_components.cu)，许可证、源码与二进制哈希均保存。没有替换 MoRGS 自定义 SAM2 Python 包。

工作副本 `working/MoRGS` 只有两处源码修改：掩码步长与 Windows 分隔符。原始 3290 个归档文件逐一复核；工作副本因导入／编译而重建的 pyc 和 egg-info 另列。新增二进制及脚本独立记录；当前运行的是含工程修正的版本，不能称为未修改作者基准的逐位复现。

## 教师权重与真实帧推理

SEA-RAFT [官方 HF 模型](https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/tree/eb97ef34ba5d856c3fa2cdcd073150c057ac8b69) 固定 revision `eb97ef34ba5d856c3fa2cdcd073150c057ac8b69`，safetensors SHA256 与 resolver 的 X-Linked-ETag 一致：`cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102`。

strict load 首次暴露共享 BatchNorm 重复名称缺失。源码确认 `bn3 is downsample[1]`，补齐 20 个同张量别名后，472 个 state keys 严格加载通过；452 个原始键及全部张量保持逐元素一致，pth 再加载也一致。没有随机填补参数或接受静默缺权重。

SAM2 使用 [官方 sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)，898083611 字节；本轮本地 SHA256 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`，不冒称官方发布校验值。ResNet34、VGG16、LPIPS 线性权重也已缓存并锁定。

在真实训练相机 `coffee_martini/cam02` 第 0／1 帧、676×507 上，实际上游 helper 执行：

- SEA-RAFT 双向 flow/info/heatmap 有限，flow 为 1×2×507×676；同帧流均值范数约 0.00738 像素。
- SAM2 的 init_state3、矩形提示、两帧传播成功；输出尺寸正确。
- eval/no_grad，保持 vendored 输入与配置路径，没有改预处理或插值定义。

SAM 矩形仅是健康检查提示，没有建立运动教师缓存。vendored SAM 预处理说明提到 mean/std，但实际返回值未归一化；阶段 02 须区分作者行为和统一教师协议并核验质量。本轮有限输出不代表教师质量或缓存回放已经通过。

本轮构造与最小调用：SEA-RAFT 约 4.00 秒、peak allocated 162221568 字节；SAM2 约 4.52 秒、1659312128 字节。精度见 JSON。它们不是场景训练显存／耗时。结束时 GPU 健康，nvidia-smi 约 6603 MiB 桌面／外部使用，不能与 allocated 混算。

## 入口与评估适配

train.py／metrics_video.py 的 --help 在目标环境加载成功；真实 GaussianModel 前后向通过。没有将不存在的 render.py 作为入口。场景图像仍由 training_report 保存，任意帧独立恢复渲染尚待后续验证。

原 metrics_video.py 对两张 128×128 合成图运行：解析 PSNR 48.1308036 dB，实际 48.1308022 dB，SSIM／LPIPS 有限。`smoke/metrics_contract` 是环境 fixture，不是性能结果。

新 `scripts/evaluate_streaming.py` 逐帧送 GPU、复用 LPIPS，保持原度量及聚合；与原入口逐帧 PSNR／SSIM／LPIPS 差异全部为 0。输出独立 eval_streaming 目录。完整序列仍需检查帧数、文件配对及尺寸。

实际命令见 `phase01_recovery_commands.md`。激活使用 scripts/run_official.cmd；教师 checkpoint 与 `configs/sam2.1/sam2.1_hiera_l.yaml` 显式传入。

## 下一阶段

阶段 02 建立独立 dynerf 数据树、冻结划分、train-only 点云、重投影及教师缓存。六场景原视频保留，Meet Room 尚未下载。流尺度／网格、offset 与 residual、密度身份及完整 checkpoint 问题仍按方案处理；软件恢复不等于这些研究问题已解决。数据和相机验证完成前不启动完整场景训练。
