# 基础 PPO 来源与维护

基础 PPO 从本机 `/home/sunteng/projects/legged_wbc_mjlab/rsl_rl` 提取，参考仓库提交为 `0f7d22cab7f4a3f82cf5fe346aa306815ea9b2f6`，包元数据为 `rsl-rl-lib==5.4.2`，上游项目为 [RSL-RL](https://github.com/leggedrobotics/rsl_rl)。参考目录在迁移时没有源码修改；其中缓存未迁入。

本目录由 quadruped_lab 独立维护，移植后使用 `agents.base` 命名空间，不读取参考仓库。保留来源代码的 ETH Zurich、NVIDIA 和归一化模块 Preferred Networks 版权通知；项目适配使用文件中的 2026–2027 zh 声明。来源代码按 [BSD-3-Clause 许可](LICENSE) 保留，根目录项目许可不替代此处第三方许可。LICENSE 原文来自本项目已安装的同版本 RSL-RL 发行包；末尾提到的依赖许可可在各自安装的发行包中查看，本目录不打包这些依赖。

主要调整：删除 HIM、蒸馏、RND、对称增强、循环/CNN 网络、多 GPU、编译、混合精度和导出；保留前馈模型、GaussianDistribution、观测归一化、RolloutStorage、PPO 和 OnPolicyRunner。日志改为英文控制台与可选 JSONL；修正环境复用观测缓冲区时的采样记录、不能整除时的样本遗漏、单样本优势归一化，以及重复 learn/恢复时的迭代计数。

基础模型的参数和归一化字段沿用来源命名，但不保证扩展模型或来源 runner 检查点兼容：本项目 `iter` 表示已完成更新次数。项目代码导入本地 `agents`；MjLab 的发行依赖 `rsl-rl-lib` 仍由 uv 安装，两者命名空间独立。
