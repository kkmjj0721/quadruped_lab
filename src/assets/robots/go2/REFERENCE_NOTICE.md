<!-- Copyright (c) 2026-2027 zh -->

# Go2 参考资产来源

`xmls/go2_reference.xml` 来自 [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a/go2_supermimic/robot)，固定提交 `d9eee9b3f248e5300e636c433255db2c1c3e086a`。该项目基于 MuJoCo Menagerie Go2；保留其 [BSD 3-Clause 许可](LICENSE_menagerie_go2) 和参考项目 [MIT 许可](LICENSE_go2_supermimic)。本地只翻译 XML 说明，模型物理参数、碰撞基本几何与 site 定义保持原样。

模型使用的16个网格与现有 `xmls/assets/` 文件逐个 SHA-256 相同，因此共享文件。新模型关节顺序为 FL、FR、RL、RR，各腿 hip、thigh、calf；参考状态代码原先的 FR、FL、RR、RL 腿排列在任务读取处明确重排。新模型的根 body 名为 `base`，旧模型为 `base_link`，分别装配，不能混用索引。

`../go2_reference.py` 和集中配置从参考 `robot/go2.py` 移植电机惯量、PD 增益、力矩及速度限制、动作比例、0～2物理步延迟与碰撞材料。延迟队列没有全部显式进入世界模型，这沿用参考的近似，不代表确定性模型能精确复现隐藏队列。
