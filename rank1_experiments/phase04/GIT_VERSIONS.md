# 阶段04 Git版本记录

根目录 `E:\4D_Gaussian` 是本地实验仓库。没有配置远端或推送。代码、配置、补丁、测试和报告随版本提交；大数据、环境、缓存和 checkpoint 留在磁盘，使用产物索引中的 SHA256 追溯。

| 提交 | 版本与用途 |
|---|---|
| `5f00cf1` / `phase03-validated` | 阶段03核验后的原协议快照 |
| `89bcbfa` | v1：源 alpha*T、CPU参考、共同损失 |
| `59c12ac` | v2：实际几何训练适配、UID/offset修正、共同K64 |
| `1588fdd` | v3：与原RGB一致的完整投影、空几何、独立评估 |
| `c29df19` | 独立评估入口生成 |
| `53852ef` | v4：密度变化后覆盖诊断、运行时源码哈希 |
| `db27fe0` | 逐帧身份保存和恢复诊断 |
| `6b0594b` | 真实像素精确参考及相机深度测试 |
| `d1af4ec` | 保留旧B1产物、协调串行GPU核验 |
| `cec883a` | v5：共同K64默认值与可复跑交付工具 |
| `1f3d354` | 明确弃权原因，保持固定教师有效像素分母 |

最终交付提交由标签 `phase04-validated` 定位。运行时版本见 `manifests/phase04_commands.jsonl` 中的 `git_commit` 与 `code_sha256_at_start`。早期运行没有入口模块哈希时，保留原Git上下文，不倒填版本。

最终比较中的所有方法采用相同完整投影和共同K64。v2的coffee B1已归档并从同一初始化重跑；后续诊断字段的变化不改变候选或损失计算。弃权原因诊断的最终版也已通过CPU/GPU测试。

小产物完整锁见 `manifests/phase04_artifact_lock.json`；44个主实验、恢复和旧版本checkpoint的文件大小与完整SHA见 `manifests/phase04_checkpoint_index.json`。最终比较表不包含旧版本产物。
