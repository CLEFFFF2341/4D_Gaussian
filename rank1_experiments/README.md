# 当前工程：干净在线基线

训练入口 `working/MoRGS/train.py` 已转接到 `../morgs_clean_baseline/train.py`。
保留 MoRGS 在线工程骨架，关闭 flow supervision、learned offset 和 motion confidence/SAM2 三个模块。
模型只有六属性，目标仅 RGB L1 + DSSIM，不带任何之前的自定义算法。
构建、使用方法和测试证据见 `../morgs_clean_baseline/README.txt`。

旧实验代码已移出当前项目，旧队列不可恢复执行。数据和历史非代码记录保留。
当前状态：CLEAN_BASELINE_VERIFIED_STOPPED。没有正式训练运行。
