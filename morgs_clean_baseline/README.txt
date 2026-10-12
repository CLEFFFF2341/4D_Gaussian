干净基线（morgs_removed_three_v1）

保留 MoRGS 在线逐帧残差工程骨架、普通 Gaussian 克隆/分裂/剪枝和原属性编码器。
关闭光流监督、learned offset_3d、motion confidence/SAM2 与其残差门控。
模型只有 xyz、f_dc、f_rest、sc、rot、op 六种属性；目标仅 RGB L1 + DSSIM。
没有自定义算法、位置码流优化器、额外分支或片元网络。
这是去三模块的在线 MoRGS 工程基线，不等同于 CVPR 2024 原版 4DGS。

入口
  morgs_clean_baseline/run.cmd train.py --scene coffee_martini --frames 90
  rank1_experiments/scripts/run_official.cmd train.py --scene coffee_martini --frames 90
两者都进入同一干净基线。训练需要已有 frame_0001.pt，或先显式执行 --bootstrap。
本次清理没有启动正式训练。

独立构建与小测试
  morgs_clean_baseline/build.cmd
  morgs_clean_baseline/run.cmd smoke_test.py
Windows 脚本使用现有 envs/morgs-cu121、VS2019 和 RTX4090 / CUDA 12.1。
独立 clean_rasterization 扩展避免加载环境内历史同名扩展。它只导出普通 rasterizer 接口。
保留三项通用修正：8列 update_mask 的正确步长；冻结图元仍更新反向透射递推；
alpha=.99 截断处的正确导数；Python backward 保存每点 motion 输入以匹配 ABI。
这些修正不引入实验算法。兼容 ABI 的 flow/motion 输出始终关闭，没有可学习参数。

验证
两场景的第90帧渲染与历史无三模块基线图像一致，最大差异0.001960814（半个8位量化单位）。
两场景各执行一次真实 RGB 反向传播与优化更新，六属性梯度均有限。
coffee 的第1->2帧残差切换执行两步真实更新，检查点重新加载后渲染逐元素相同。
测试不访问 cam00；使用训练视角及 cam01 验证视角。
backbone_checks.json、source_audit.json 和 cleanup_manifest.json 保存检查证据。

版本管理
源码、配置、构建脚本和小测试证据进入 Git；数据、权重、初始化/参考检查点、
新测试检查点、编译产物及 Python 字节码不进入 Git。大型数据继续留在本地。
递归删除被自动审批拒绝（仅返回 blocked by policy）。旧代码改为移出项目目录，
保存在 E:/4D_Gaussian_retired_code_20261012，不属于当前工程或 Git 提交内容。
