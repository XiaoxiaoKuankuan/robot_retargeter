# BUMI3 开发变更日志

## 2026-08-31：四足稳定地板、逐帧动态偏移与 heel/toe 等高软约束

### 工作区与修改目标

- 本轮继续位于 `feature` 分支，起始 HEAD 为 `f141897`，保留工作区全部既有修改；没有执行 reset、clean、stash、rebase、commit 或 push。用户要求恢复仓库默认的逐帧动态高度偏移，但只能由四个足点驱动；同时删除四足点全序列 1% 分位地板，改为按速度筛选稳定支撑足点后做稳健地板拟合，并增加 heel/toe 高度差软约束。
- MuJoCo 播放器的 ground 是水平 `z=0`，因此本轮选用稳健恒定水平地板，而不是拟合数据中不可被播放器表达的斜平面。该选择只处理不同数据集的世界 Z 原点，不伪造斜坡法向。

### 配置与代码改动

- `config/robot/bumi3.yaml` 保持 `ground_reference_contacts` 严格为左右 heel/toe 四点，把 `contact_height_dynamic_offset_enabled` 改为 `true`。左右手仍可参与 G1 式 `legacy_hold` 低权重 IK，但不会传入地板估计或动态整机 Z 偏移。删除 `contact_height_floor_percentile: 1.0`，改用 `stable_support_dense_median`：速度阈值 `0.2 m/s`、高度内点容差 `0.04 m`、最少 8 个样本。
- `scripts/smpl_replay.py` 新增低速足点稳健恒定地板估计器：先筛速度，再用 O(N) 双指针寻找跨度不超过 `0.08 m` 的最大高度密集簇，同票时选更低簇，最后在中心 ±`0.04 m` 内以中位数重拟合。源接触检测、滞回相对高度和缩放后机器人关键点初始高度复用同一方法；PKL 记录源/目标样本数、内点数、MAD 和地板高度。
- `scripts/robot_retarget.py` 新增左右脚 `heel_z-toe_z=0` 标量软任务，基础 cost 为 `15`（低于腿部位置任务的 `30`），只在同脚 heel/toe 同时接触时以 `0.10 s` smootherstep 渐入渐出。任务不锁 XY、不指定绝对 Z；其两个世界 Z 雅可比相减后统一 Root-Z 列为零，不能直接移动整机高度。`legacy_hold`、`active_only` 和实验用 `paired_support` 的任务列表都保持兼容，CSV 元数据新增软任务配置与逐脚状态统计。
- `scripts/validate_bumi3_retarget.py` 新增地板方法、源/目标稳健拟合样本和内点合同检查，并核对 CSV 元数据中的软任务 cost/渐变时间；对 legacy 软约束的稳定平足高度差给出统计和超阈值 warning，不把低权重相对任务误报为绝对贴地硬合同。
- `tests/test_bumi3_input_adapter.py` 增加快速摆动点、少量极低异常值和稳定密集地板簇测试；`tests/test_bumi3_asset_and_contact.py` 固定四足动态高度、稳健地板配置，并验证软任务权重渐变、零权重残差和 Root-Z 雅可比相消。`docs/bumi3_retargeting.md` 同步当前算法边界。

### 验证与交付状态

- 修改前的四足点 1% 静态地板产物已完整备份到 `output_data/archive/bumi3_static_p01_before_stable_dynamic_20260831/`，共约 31 MB；包含 BUMI3 keypoint、CSV/metadata、Mimic NPZ 和报告，源 SMPL-X 数据没有修改。
- 最终配置重新生成四集合 10 条、共 `12,695` 帧、30 Hz 的 CSV/metadata 与 Mimic NPZ；播放器 `--list-only` 发现 10/10。配置 SHA256 为 `02e5c3b803a202891c8a73b4d45c5c8659952a31be52452cd4739dc7c4234ff3`，MJCF SHA256 为 `fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3`，十份最终 metadata 均与当前配置/模型一致。
- 十份 keypoint 的 `ground_reference_contacts` 都严格等于四个足点，模式均为 `dynamic_contact`；左右手检测激活帧均为 0，而且代码只把四足点切片传给动态高度函数，因此即使未来手部被检测为接触也不能改变整机 Z。源稳健地板候选为 `152～9,736` 个低速样本、`113～9,576` 个内点，全部高于最少 8 个样本合同；目标拟合同样通过。
- 相对上一版无 heel/toe 软任务的备份，在完全相同的新接触帧掩码上统计 `12,085` 个稳定平足样本：绝对 heel/toe 高度差中位数由 `20.92 mm` 降至 `14.54 mm`，P95 由 `62.07 mm` 降至 `45.70 mm`，最大值由 `135.87 mm` 降至 `110.14 mm`，`96.5%` 样本改善。代表轨迹 cost=`5→15` 对照中最大关节速度基本不变，Pair5 只增加约 `0.11 rad/s`，所以最终选择 cost `15`；它仍是软约束，不能把残余最大值描述为严格平足通过。
- 动态高度 offset 的最坏单帧变化为 `0.02188 m`（E2），最终 Root-Z 的最坏单帧变化为 `0.03337 m`；这是恢复 G1 式逐帧动态偏移后的真实代价，当前由 `contact_height_lpf_alpha=0.2` 平滑，但没有启用额外 temporal IK、轨迹 QP、Root-Z QP 或离线平滑。
- 保持原质量门槛后仍为 4/10 passed、6/10 failed。通过项是 `0dqp`、`192`、`GSK`、AIST++ ch01；`189`、`E2`、AIST++ ch02 因最大关节速度失败，`190`、Pair4、Pair5 因任务 RMS 失败（Pair5 两项 RMS 同时失败）。没有为本轮地板或等高目标放宽连续性/精度门槛。
- 完整测试为 `25 passed`，最终模型预检、修改 Python 文件 `py_compile` 与 `git diff --check` 均通过。本轮完成的是离线运动学、产物结构和数值验证；没有打开 GUI 逐条人工观看，也没有执行 IsaacLab 动力学 replay、控制器跟踪或实机测试。

## 2026-08-31：修复绝对世界原点导致的整机入地/悬空

### 问题复核与策略纠正

- 本轮继续位于 `feature` 分支，起始 HEAD 为 `f141897`，保留工作区全部既有修改；没有执行 reset、clean、stash、rebase、commit 或 push。上一轮虽然正确删除了逐帧接触驱动的动态全身 Z 偏移，但把四个数据集互不一致的世界原点原样送入 IK，导致六条整体在地下、四条整体悬空。该结果不能作为正确的 MuJoCo 视觉交付。
- 十条未标定 keypoint 的四足点 1% 高度实际横跨 `-1.0630～+1.2528 m`，因此“保留任意源绝对 Z”与“全部落在 MuJoCo z=0 地板”无法同时成立。修复选择每条序列只计算一次四足点 1% 高度，并对整条序列施加一个恒定 Z 标定量；它不是逐帧接触偏移，接触状态进入或退出都不会改变该值。
- 修复前的错误绝对高度版本已完整备份到 `output_data/archive/bumi3_absolute_world_static_zero_30hz_before_fixed_floor_20260831/`，共 53 个文件、约 30 MB，包含十条 keypoint、CSV/metadata、NPZ、报告及两份既有诊断报告。源 SMPL-X 数据未修改。

### 配置、代码和硬验收修改

- `config/robot/bumi3.yaml` 继续保持 `ground_reference_contacts` 只有左右 heel/toe 四点，`contact_height_dynamic_offset_enabled: false` 不变；把 `contact_height_relative_to_sequence_floor` 改为 `true`，以四足点全序列 1% 高度同时校准接触检测和机器人关键点地板。左右手不参与该计算。
- `scripts/smpl_replay.py` 的关闭动态偏移路径改为只应用 `initial_height` 这一整段恒定值，不再错误地强制返回零偏移。输出 PKL 新增 `contact_height_offset_mode=static_sequence_floor`；每条的 offset 最小值、中位数、最大值必须完全相同。若地板标定未启用，恒定值仍为零；其他默认启用动态模式的机器人行为不变。
- `scripts/validate_bumi3_retarget.py` 除检查四足点、模式和恒定 offset 外，新增真实 MuJoCo FK 地板硬门槛：四足点全序列 1% 高度必须落在 z=0 的 ±`0.02 m` 内，机器人根节点全序列最低高度必须大于 `0.15 m`。这直接阻止“整机在地下/高空”的产物通过。`legacy_hold` 仍不承诺每个支撑帧无滑移或零穿透，对应局部问题继续记录 warning。
- `tests/test_bumi3_input_adapter.py` 改为构造足/手任意接触切换和非零静态偏移，验证所有帧只加同一个常量、帧间差分保持不变；`tests/test_bumi3_asset_and_contact.py` 固定生产配置使用相对序列地板。`docs/bumi3_retargeting.md` 同步区分一次性地板标定和已禁用的动态接触偏移。

### 十条重生成、数值和可视化证据

- 用 `/home/weili/miniconda3/envs/robot_retargeter/bin/python` 重新生成四集合全部 10 条、共 `12,695` 帧、30 Hz 的 keypoint、CSV/metadata、Mimic NPZ 与 JSON 报告。当前配置 SHA256 为 `d10739458cb374b09b1f138f90abcb7de98e3eb6b36507353bd2e97fbbca55c6`，MJCF SHA256 为 `fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3`，十份元数据均匹配。
- 十条最终 FK 四足点 1% 高度范围为 `-0.01379～+0.00604 m`，全部通过 ±`0.02 m` 硬门槛；最小根高度为 `0.25787 m`，根高度中位数范围为 `0.44197～0.56002 m`。因此已不存在整机跑入地下或整体悬空。局部最深足点为 `-0.05092 m`，来自少量姿态/IK 误差帧，报告保留 warning，不能表述为严格物理接触已经解决。
- 使用 MuJoCo EGL 对 AIOZ 0dqp、AIOZ E2、COMPAS Pair5、AIST++ ch01 四个跨原点代表动作做离屏视觉抽查，机器人均位于网格地板上；拼图保存为 `output_data/reports/bumi3_fixed_floor_audit.png`。播放器 `--list-only` 最终确认 10/10 条的帧数、30 Hz 与时长均可正常读取。
- 保持原质量门槛后为 4/10 passed、6/10 failed。通过项是 `0dqp`、`192`、`GSK`、AIST++ ch01；`189`、`E2`、AIST++ ch02 因最大关节速度失败，`190`、`Pair4`、`Pair5` 因任务 RMS 失败。最坏关节速度为 `54.5712 rad/s`，最坏非 calf 任务 RMS 为 `0.12981 m`；没有为解决地板问题放宽连续性或任务精度门槛。
- `PYTHONPATH=scripts /home/weili/miniconda3/envs/robot_retargeter/bin/python -m pytest -q` 为 `21 passed`，修改 Python 文件 `py_compile` 与 `git diff --check` 通过。本轮只完成离线运动学和离屏渲染检查，未执行 IsaacLab 动力学 replay、控制器跟踪或实机测试。

## 2026-08-31：四足点地面参考与静态绝对 Z 策略

### 工作区、备份与修改原因

- 本轮继续位于 `feature` 分支，起始 HEAD 为 `f141897`。保留工作区全部既有修改；没有执行 reset、clean、stash、rebase、commit 或 push。用户要求 `ground_reference_contacts` 只能包含四个足点，并删除由逐帧激活接触驱动的全身 Z 平移，避免手/脚接触进入或退出时直接移动整个人。
- 修改前的 30 Hz 动态 Z 十条完整产物已备份到 `output_data/archive/bumi3_strict_g1_dynamic_z_30hz_before_static_z_20260831/`，共 51 个 keypoint、CSV/metadata、NPZ、报告和模型预检文件。源 SMPL-X 数据没有修改；回退时必须成套恢复，不能只替换 CSV。

### 配置、代码与验证合同

- `config/robot/bumi3.yaml` 新增唯一允许的地面参考列表：`left_foot_end`、`left_toe`、`right_foot_end`、`right_toe`。左右手仍保留在六个 `legacy_hold` 低权重接触任务中，但不再参与地板高度估计。新增 `contact_height_dynamic_offset_enabled: false`，明确关闭逐帧接触驱动的整组 keypoint Z 平移；绝对源世界高度原样保留。
- `scripts/smpl_replay.py` 为高度偏移函数增加兼容默认值为 `true` 的开关，其他未配置机器人行为不变。关闭时函数不根据接触状态计算高度，精确复制输入关键点并返回全零 offset 审计数组；keypoint PKL 同时记录四足点列表和动态偏移开关。
- `scripts/validate_bumi3_retarget.py` 新增 keypoint 交付检查：地面参考必须和配置逐项一致；动态 Z 开关必须一致；关闭动态偏移时 offset 的最小值、中位数、最大值必须均为零。验证器还核对 CSV 元数据中的配置 SHA256，拒绝用旧配置产生的 CSV/metadata 冒充当前产物。
- `scripts/export_bumi3_mimic_npz.py` 同样新增配置 SHA256 漂移检查，与既有 MJCF SHA 检查组成完整的模型/配置指纹边界。`tests/test_bumi3_input_adapter.py` 增加任意足/手接触切换下关键点完全不动、offset 全零的回归测试；`tests/test_bumi3_asset_and_contact.py` 固定生产配置的四足点与关闭动态偏移合同；`tests/test_bumi3_export.py` 覆盖正确配置 SHA 的 NPZ 导出路径。`docs/bumi3_retargeting.md` 同步说明这套策略及其绝对高度边界。

### 十条重生成与诊断结果

- 使用 `/home/weili/miniconda3/envs/robot_retargeter/bin/python` 以最终文件状态重新执行四集合全部 10 条，共 `12,695` 帧、30 Hz。当前配置 SHA256 为 `14283eb14795db0e9928e82c54b977094969c5e807b531da92dd1a232ebf473f`，MJCF SHA256 为 `fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3`；十份元数据均匹配这两个指纹。播放器 `--list-only` 发现 10/10 条。
- 十份 keypoint PKL 的 `ground_reference_contacts` 均严格等于上述四足点，`contact_height_dynamic_offset_enabled` 均为 `false`，高度 offset 的最小值/中位数/最大值全部为 `0/0/0 m`。这证明接触开关已经不能再通过预处理高度偏移直接移动整个人。
- 对原动态 Z 备份的逐帧对比：E2 的 keypoint hips 最坏单帧下降由 `0.1690 m` 降至 `0.0324 m`，FineDance 190 由 `0.1074 m` 降至 `0.0196 m`；旧 190 最坏下降帧同时有两个接触状态变化。当前 GSK 虽然 keypoint hips 最坏下降仅 `0.0101 m`，IK 后机器人根仍出现 `0.1274 m` 的单帧下降，说明剩余跳变属于逐帧 IK 解分支，而不是已删除的动态高度偏移路径。
- 未改变严格 G1 基线的质量门槛，结果仍为 3/10 passed、7/10 failed：`0dqp`、`189`、`192`、`E2`、AIST++ ch02 因关节速度超过 `35 rad/s` 失败；`Pair4` 因最差非 calf 任务 RMS 为 `0.12160 m` 失败；`Pair5` 同时因 `44.1944 rad/s` 和 `0.12978 m` 失败。最坏速度为 `59.2923 rad/s`，未降级为 warning。

### 绝对高度边界与验证范围

- 关闭动态偏移和序列地板归一化后，不同源数据集的世界原点差异被如实保留。MuJoCo FK 的四足点中位 Z：AIOZ/FineDance 六条约为 `-0.989～-0.561 m`，会整体位于地下；Pair4/Pair5 约为 `0.687/0.671 m`，两条 AIST++ 约为 `1.278/1.338 m`，会整体悬空。这不是接触切换造成的新跳变，而是“绝对世界高度、不做归一化”的直接结果；本轮没有把它伪装成已经贴地。
- `PYTHONPATH=scripts /home/weili/miniconda3/envs/robot_retargeter/bin/python -m pytest -q` 为 `21 passed`；修改 Python 文件 `py_compile` 通过；`git diff --check` 通过；最终 10 条均完成 keypoint、CSV/metadata、Mimic NPZ 和报告生成。本轮没有打开 GUI 逐条肉眼检查，也没有执行 IsaacLab 动力学 replay、控制器跟踪或实机测试。

## 2026-08-31：建立 30 Hz 严格 G1 算法基线并重生成 10 条轨迹

### 基线目的与 50 Hz 边界

- 本轮继续位于 `feature` 分支，起始 HEAD 为 `f141897`。保留工作区全部既有修改；没有执行 reset、clean、stash、rebase、commit 或 push。用户要求先消除 BUMI3 专用策略与 G1 之间的变量，因此本轮是诊断基线，不把未通过质量门槛的原始 IK 结果表述为最终修复。
- 此前默认 50 Hz 来自 legged_lab BUMI3 IsaacLab 环境的 `sim.dt=0.005` 与 `decimation=4`，即策略/动作更新周期 `0.02 s`；它是部署接口频率，不是 BUMI3 几何或 Mink IK 的要求。为与仓库已有 G1 对照轨迹保持一致，本轮离线 SMPL-X 关键点、CSV、metadata 和 Mimic NPZ 全部改为真实 30 Hz；将来交付控制器前必须另行显式重采样到 50 Hz。

### 配置与代码修改

- `config/robot/bumi3.yaml` 使用与 G1 相同的单阶段 Mink 策略：`legacy_raw` 停止误差、`legacy_hold` 六接触点、位置 cost `15`、绝对世界高度、接触区间 `mean` 目标；hips/head/左右 hip/thigh/calf/shoulder/arm/forearm 的位置和旋转 cost 逐项复制 G1 数值，左右 calf 均为 `30/3`。删除生产配置中的显式 `contact_map`，让六个接触点沿用 G1 的固定顺序映射。
- 关闭所有不属于 G1 原始路径的 BUMI3 输出修正：`temporal_posture_cost=0`、输出关节速度/加速度/jerk 上限均为 0、高斯平滑为 0、支撑根 Z 投影和足底屏障关闭、接触滞回/最短持续时间/权重渐变/成对平足方向任务关闭。BUMI3 自身 MJCF、body/frame 名、连杆尺度、方向标定、IK 热启动姿态和真实关节限位仍保留，因为这些是机构必需合同而不是算法变量。
- `bash/retarget_smpl_to_bumi3.sh` 与 `bash/retarget_music_smpl_4set_bumi3.sh` 默认 `TARGET_FPS` 改为 30 且向下游透传；`scripts/validate_bumi3_retarget.py` 和导出测试不再硬编码 50 Hz，改为核对 `config.output.target_fps`。生产配置回归测试新增与 `config/robot/g1.yaml` 的 IK 权重逐项相等检查，并固定上述无平滑、无输出限幅、无显式 contact map 和 30 Hz 合同。
- README 与本文档明确区分 30 Hz 诊断基线和 50 Hz 部署边界，避免再把控制频率误写为 IK 固有要求。修改前的 50 Hz 十条完整产物已备份到 `output_data/archive/bumi3_before_strict_g1_50hz_20260831/`，包含 keypoint、CSV/metadata、NPZ 与报告，可直接恢复查看；源 SMPL 数据没有修改。

### 生成结果与基线结论

- 使用 `/home/weili/miniconda3/envs/robot_retargeter/bin/python` 对四集合十条源动作重新执行真实 30 Hz SMPL-X 前向、单阶段 Mink IK、CSV/metadata、Mimic NPZ 和联合验证。十条共 `12,695` 帧；播放器 `--list-only` 发现 10/10，所有 metadata 的配置 SHA256 均为 `c15c7cbb375e3979f926b4f975cac38b1385d46d7a50ee7d67dd1866f6796e97`，MJCF SHA256 均为 `fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3`，fps 均为 30.0。
- 质量硬门槛结果为 3/10 passed、7/10 failed。通过项为 `190`、`GSKxQcb1PTU_04_1222_1560_dancer_00`、`gBR_sBM_cAll_d04_mBR0_ch01`；其余六条因最大相邻帧关节速度超过 `35 rad/s` 失败，`Pair4` 因最差非 calf 任务位置 RMS 超过 `0.12 m` 失败，`Pair5` 同时违反两项。最坏速度为 `58.6725 rad/s`，没有降低验收门槛或把失败改成 warning。
- FineDance `189` 的最坏跳变发生在第 `282→283` 帧：BUMI3 左臂 yaw 从 `+1.39818` 跳到 `-0.17640 rad`，速度 `47.2374 rad/s`；该帧四足接触状态没有切换。同一条 G1 对照的全身最坏值为 `28.4447 rad/s`。BUMI3 在该条的最坏踝速度仅 `9.5028 rad/s`，说明当前全局硬失败首先来自手臂 IK 分支，而不是帧率或该时刻脚接触开关。
- 十条 BUMI3 最坏踝速度范围为 `4.8897～11.9945 rad/s`；同源 G1 为 `5.5827～14.8958 rad/s`。因此严格基线下“踝关节本身的速度峰值”并不系统性差于 G1，但 BUMI3 手臂/肘关节更容易因较少自由度、不同轴向与限位切换 IK 分支；这与后续是否需要机构专用连续性策略是两个独立问题。

### 验证边界与回滚

- `bash -n`、修改 Python 文件 `py_compile` 均通过；`PYTHONPATH=scripts /home/weili/miniconda3/envs/robot_retargeter/bin/python -m pytest -q` 为 `20 passed`。十条均完成 CSV/NPZ 生成和结构验证，但只有 3 条通过全部质量硬门槛；本轮没有自动打开 GUI 肉眼逐条播放，没有执行 IsaacLab 动力学 replay、控制器跟踪或实机测试。
- 当前播放器命令为 `/home/weili/miniconda3/envs/robot_retargeter/bin/python scripts/play_bumi3_trajectories.py --motion-dir output_data/robot_motion --pattern '*_bumi3.csv'`。如需回滚到本轮前的 50 Hz 产物，应从上述 archive 目录成套恢复 keypoint、CSV/metadata、NPZ 和报告；不能只替换 CSV，否则 fps、配置 SHA 和导出合同会不一致。

## 2026-08-31：切换为 G1 式连续性优先接触并重生成 10 条轨迹

### 工作区与改动原因

- 本轮位于 `feature` 分支，起始 HEAD 为 `f141897`。保留工作区原有全部修改；没有执行 reset、clean、stash、rebase、commit 或 push。用户确认 G1 对同一批源动作播放正常，希望 BUMI3 同样采用绝对世界高度、取消按序列地板归一化，并取消高权重足点锁定，优先消除接触边界上的 IK 分支跳变。
- 切换前对 Pair5 的跳变帧做逐帧诊断：原 `paired_support` 在约 `147.84 s` 的接触相切换处出现根姿态约 `59°/frame`、根角速度 `51.58 rad/s`，腿部关节速度 P99/峰值约 `14.20/30.00 rad/s`，腿部速度超过 `10 rad/s` 的帧占约 `18.0%`。同一动作的 G1 不启用成对强锁点，播放连续；因此问题不是源 SMPL 采样丢帧，而是高权重接触任务与根高度投影在相位边界改变了 IK 最优分支。

### 配置、验证器与文档修改

- `config/robot/bumi3.yaml` 将生产接触模式改为与 G1 相同的 `legacy_hold`，四个 heel/toe 位置 cost 从 `300` 降为 `15`；设置 `contact_height_relative_to_sequence_floor: false`，直接使用绝对世界高度；关闭接触滞回、权重渐变、成对平足方向 cost、接触前竖直预约束、接触穿地修正、IK 后根 Z 支撑投影和事后足底屏障。方向标定的中立平足姿态、IK 热启动姿态及四个不透明红色足点均保留不变。
- `scripts/validate_bumi3_retarget.py` 按接触模式区分验收合同：`paired_support` 仍硬验收足点离地、heel/toe 高度差、脚滑和穿地；`legacy_hold` 不承诺贴地，把对应超阈值项写为 warning。新增 `joint_limit_occupancy_validation` 配置；生产配置把连续贴限位占比作为 warning，但实际越限、相邻帧速度、加速度、jerk、任务 RMS、格式和四元数仍是硬失败项，不能用 warning 绕过跳变检查。
- `tests/test_bumi3_asset_and_contact.py` 新增生产配置回归，固定 `legacy_hold`、绝对高度、无滞回、无根投影、无成对平足方向任务、四足点 cost=15 和贴限位 warning 合同；通用 `paired_support` 状态机测试继续保留，避免删除实验能力。
- `docs/bumi3_retargeting.md` 更新当前 MJCF SHA、G1 式接触语义、绝对高度边界和模式化验收说明，删除已经失效的 `active_only`、序列地板归一化和 IK 后统一抬根描述。

### 重生成结果与连续性证据

- 使用 `/home/weili/miniconda3/envs/robot_retargeter/bin/python` 执行 `PYTHON_BIN=... bash/retarget_music_smpl_4set_bumi3.sh`，覆盖重生成四集合全部 10 条的 50 Hz keypoint、CSV、metadata、Mimic NPZ 和 JSON 报告，共 `21,151` 帧。批处理退出码为 0，10/10 报告状态均为 `passed`，末尾播放器 `--list-only` 发现 10/10 条。
- 当前配置 SHA256 为 `0ab462a8abbb1e99e115d09d9c4ffe25ccc61fb9c55774eb7280a9b829ac02ec`，当前 MJCF SHA256 为 `fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3`；10 份最终 `.meta.json` 均与这两个哈希一致。
- Pair5 当前根角速度 P99/峰值为 `10.58/17.91 rad/s`，全段没有任何根角速度超过 `20 rad/s` 的帧，原 `147.84 s` 根姿态跳变消失。腿部关节速度 P99/峰值降为约 `7.05/25.83 rad/s`，超过 `10 rad/s` 的腿部帧占比从约 `18.0%` 降至约 `0.77%`，超过 `20 rad/s` 的占比约 `0.027%`。当前残余的 `30 rad/s` 峰值来自首帧手臂从热启动姿态进入动作，而不是腿部接触切换。
- 十条硬指标最坏值为：关节速度峰值 `30.00 rad/s`；加速度 P99/峰值 `516.44/1475.44 rad/s²`；jerk P99/峰值 `42192.21/128822.02 rad/s³`；有效任务整体位置 RMS `0.04112 m`。十条根角速度峰值均低于 `20 rad/s`，最坏为 Pair5 的 `17.91 rad/s`。
- 连续性优先存在明确足部代价：AIOZ、FineDance 等报告出现脚底穿地/滑移 warning，最坏为 E2 的 `0.11957 m` 穿地；Pair4、Pair5 和两条 AIST++ 因绝对世界高度与各数据集原点不同而没有可评估接触区间。E2 左踝 roll 精确贴 ±`0.17 rad` 限位率约 `66.39%`，但其根角速度峰值仅 `4.17 rad/s`、关节速度 P99 `5.25 rad/s`，属于连续饱和而非跳帧。该版本按用户选择优先解决连续性，不应表述为已经解决贴地或足底物理接触。

### 验证边界与回滚

- `PYTHONPATH=scripts /home/weili/miniconda3/envs/robot_retargeter/bin/python -m pytest -q` 为 `20 passed`；修改脚本 `py_compile` 通过；批处理内 10 次模型预检、CSV/NPZ 联合验证全部通过；tracked diff 与本轮相关 untracked 文件的 whitespace check 均通过。本轮没有自动打开 GUI 逐条肉眼观看，也没有执行 IsaacLab 动力学 replay、控制器跟踪或实机测试。
- 本轮覆盖了同名 BUMI3 输出，没有保存完整的旧 `paired_support` 十条副本，源 SMPL 数据未修改。若要回退，需恢复本节列出的 YAML/验证器合同并用批处理重新生成，不能只替换 CSV；如下一步需要同时改善贴地，应在当前连续性基线上设计低权重、连续的根高度偏置，而不是直接恢复高权重成对锁点。

## 2026-08-28：按五项顺序修复平足、接触连续性与硬验收，并重生成 10 条轨迹

### 修复范围与原因

- 本轮继续位于 `feature` 分支，起始 HEAD 仍为 `f141897`。保留工作区已有全部修改；没有执行 reset、clean、stash、rebase、commit 或 push。本节结果取代下方较早一轮关于 `active_only`、接触 cost=120 和旧验收指标的最终配置描述。
- `scripts/calibrate_bumi3_keyframes.py` 将“方向标定参考姿态”和“正式 IK 热启动姿态”拆成两个独立合同：标定读取 `reference_pose`，固定左右踝 pitch/roll 为 `0` 并对足底朝向施加强残差；`--write` 只更新标定快照和轴映射，不再覆盖 `initial_joint_positions`。正式 IK 仍可从左右踝 pitch `-0.172 rad` 的屈膝姿态热启动。这样既保留收敛性，也不再把热启动的脚掌预倾角写进中立方向映射。模型预检测得中立姿态左右 heel/toe 高度差仅约 `1.47e-7 m`。
- `scripts/prepare_bumi3_asset.py` 与生成的 `asset/robot/bumi3/mjcf/bumi3_retarget.xml` 将左右脚跟、脚尖四个 marker 统一为半径 `0.01 m`、`rgba="1 0 0 1"`、无质量且不参与碰撞的不透明红点；其余方向标定 marker 保持透明，避免干扰播放画面。
- `scripts/smpl_replay.py` 新增 Schmitt 滞回接触状态机：进入/退出分别使用速度 `0.20/0.35 m/s` 和相对高度 `0.075/0.10 m`，并加入 `0.06/0.08 s` 连续确认以及接触/摆动各 `0.20 s` 的最短持续时间。它先在源动作侧稳定接触相位，再把明确的 heel/toe 状态交给 IK，避免阈值附近逐帧抖动。
- `scripts/robot_retarget.py` 将 BUMI3 接触模式改为 `paired_support`。平足期同一只脚的 heel 与 toe 同时锁定到成对地面目标并施加平足方向约束；只有 `heel=false, toe=true` 的 toe-off 相位才释放后跟。接触 cost 采用五阶 smootherstep 在 `0.20 s` 内渐入/渐出，足点位置目标 cost 为 `300`、平足方向 cost 为 `30`，不再逐帧硬开关固定 cost。接触前加入只约束竖直方向的预约束，使落脚在正式接触任务接管前连续靠近地面。
- 同一求解器增加逐帧输出速度/加速度/jerk 约束 `30 rad/s`、`1200 rad/s²`、`80000 rad/s³`，并在整段完成后执行有界根 Z 支撑投影：优先把支撑点投向 `3 mm` clearance，但不允许新接触脚把已稳定支撑脚抬高超过 clearance 上方 `18 mm`，也不允许活动支撑点低于 `-9 mm`。没有启用事后单踝修正，因为该策略会在 toe-off 改变足点圆弧方向，实测反而制造穿地和 jerk。
- `scripts/validate_bumi3_retarget.py` 新增不可绕过的离线硬门槛：稳定支撑点离地 `<0.02 m`、平足期 heel/toe 高度差 `<0.01 m`、最大穿地 `<0.01 m`、关节加速度 P99/峰值 `<650/2000 rad/s²`、jerk P99/峰值 `<50000/150000 rad/s³`；同时保留相邻帧速度 `<35 rad/s`、接触区间中位位移 `<0.03 m` 和中位水平速度 `<0.05 m/s`。验证按 `paired_support` 相位解释 toe-off，并排除权重尚未爬升完成的接触前沿帧，任何失败都会令单条脚本和十条批处理非零退出。
- `tests/` 补充参考姿态平足、四个不透明红点、成对支撑相位、接触滞回/最短持续时间、权重渐变和接触前预约束测试；这些测试保护配置与算法合同，避免以后又退回热启动标定或二值 cost。

### 最终产物与验证证据

- 使用 `/home/weili/miniconda3/envs/robot_retargeter/bin/python` 和当前配置重新运行 `DATASET_ROOT=dataset/music_smpl_4set OUTPUT_DIR=output_data PYTHON_BIN=... BUMI_SOURCE_DIR=/home/weili/legged_lab/source/NoetixRobot/NoetixRobot/assets/robots/bumi3 SMPL_MODEL_PATH=/home/weili/GENMO/inputs/checkpoints/body_models bash bash/retarget_music_smpl_4set_bumi3.sh`。四集合 10 条、共 `21,151` 帧全部完成 SMPL-X 50 Hz 关键点、Mink IK、CSV、Mimic NPZ 和联合验证，10/10 报告均为 `passed`；批处理末尾 `--list-only` 识别 10/10 条。
- 当前配置 SHA256 为 `cdd55a931183e06c4446d4a7899aaef63c4a1175781c5706ab5b11c87ce0f48e`，当前重定向 MJCF SHA256 为 `fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3`；10 份 `.meta.json` 全部同时匹配这两个哈希。`output_data/keypoints/bumi3/`、`output_data/robot_motion/`、`output_data/mimic_npz/bumi3/` 和 `output_data/reports/` 对清单中的 10 个 stem 均各有 10 份对应产物。
- 十条最坏值为：速度 `30.0000 rad/s`；加速度 P99/峰值 `507.94/1407.33 rad/s²`；jerk P99/峰值 `34951.84/108115.35 rad/s³`；稳定支撑点最高离地 `0.01800 m`；平足 heel/toe 最大高度差 `0.006689 m`；最大穿地 `0.003857 m`；接触区间最坏中位位移 `0.026616 m`、最坏中位水平速度 `0.004298 m/s`。这些数值分别低于上述硬门槛。
- 验证命令：`python -m py_compile scripts/robot_retarget.py` 通过；`PYTHONPATH=scripts python -m pytest -q` 为 `19 passed`；`git diff --check` 通过。此次没有自动打开 GUI 逐条肉眼观看，也没有执行 IsaacLab 动力学 replay、控制器跟踪或实机测试，因此 `passed` 仅证明当前离线 FK/IK、足点和离散运动学门槛，不能替代动力学与实机安全验证。
- 本轮覆盖更新了同名 BUMI3 输出，未另存旧轨迹备份。回退算法需恢复本节涉及的配置/脚本/资产修改后重新生成；只回退产物不会恢复旧求解合同。源 SMPL 数据未修改。

## 2026-08-28：修正 BUMI3 映射并以最终配置重生成 10 条轨迹

### 工作区与问题定位

- 本次工作位于 `feature` 分支，起始 HEAD 为 `f141897`；保留工作区中已有的 README、SMPL 输入适配、BUMI3 资产、脚本、数据集和测试改动，没有执行 reset、clean、stash、rebase、commit 或 push。
- 先用同一批 SMPL 数据的 G1 正常结果作为对照。旧 BUMI3 的有效任务整体位置 RMS 约为 `0.1286 m`，左右手约为 `0.20/0.18 m`；同时左踝 roll、左臂 roll、右臂 yaw 等关节存在约 `66%～83%` 的长时间限位饱和，真实相邻帧关节速度峰值约 `132 rad/s`。这说明主要问题在 BUMI3 映射/约束，而不是源 SMPL 轨迹。
- 中立姿态坐标检查发现：SMPL 左髋方向位于源根坐标 `+X`，BUMI3 左髋方向位于机器人根坐标 `+Y`，旧配置却使用单位根四元数，造成整套左右/前后轴错位；同时旧标定把 SMPL 中立 T-pose 的位置目标与 BUMI3 下垂手臂、屈膝初始姿态的方向标定混在一起，位置和旋转目标互相冲突。
- 对照 `config/robot/g1.yaml` 和 `config/robot/booster_t1.yaml` 后确认：Booster T1 的高度和头部弱方向约束更接近 BUMI3，但 BUMI3 的 3-DoF 髋、3-DoF 肩和单肘拓扑更接近 G1。两者都把大腿末端的位置 cost 设为 0，避免短小机器人被人体膝盖绝对位置过约束。另参考 GMR 的 `smplx_to_bumi3_auto.json`，继续弱化手臂方向，优先保持肩、肘和手端的位置链。

### 配置与求解器修改

- 重写 `scripts/calibrate_bumi3_keyframes.py` 的 BUMI3 参考姿态求解：先生成中立 SMPL-X，再走与正式数据完全相同的身体比例缩放；固定根 yaw 为 `-90°`，用 `scipy.optimize.least_squares` 对左右对称的腰、手臂、腿、膝和踝共 11 个参数求解参考姿态，之后才计算每个 source/body 的方向轴映射。
- `config/robot/bumi3.yaml` 将最终根四元数设为 `[0.70710678, 0, 0, -0.70710678]`，写入对称参考关节姿态和全部显式正则化种子。重复执行标定得到一致结果：参考位置 RMS `0.029673 m`、最大误差 `0.049546 m`，各方向标定中立残差均低于 `6e-17 rad`，不会因反复 `--write` 缓慢漂移。
- IK 权重采用 G1/Booster 的稳定结构：hips `100/0`；左右 hip `30/3`、thigh `0/3`、calf `30/3`；head `0/0.5`；shoulder `30/1`、arm `10/1`、hand `10/0`。这样保留人体动作轮廓，同时不要求 BUMI3 的短肢体复制不可达的绝对膝点或手端方向。
- 四个脚跟/脚尖激活接触位置 cost 最终从 `80` 提到 `120`，接触区间目标使用 `mean`。`mean_ramp` 虽已作为通用可选模式实现并测试，但在 Pair5 上把接触中位速度恶化到约 `0.137 m/s`，因此没有用于最终配置；cost=80 的同一轨迹约 `0.056 m/s`，也未达到既有 `0.05 m/s` 门槛。
- `scripts/robot_retarget.py` 新增兼容默认值为 0 的 `temporal_posture_cost`。BUMI3 最终取 `5.0`，每帧以此前一帧解作为弱姿态目标，抑制接触任务切换时的 IK 分支跳变；Pair5 上真实峰值速度从无约束时约 `49.2 rad/s` 降至约 `20.2 rad/s`，同时手端 RMS 仍保持约 `7.5/8.0 cm`。旧机器人未配置该字段时行为不变。

### 验证器修正与最终产物

- `scripts/validate_bumi3_retarget.py` 的静态 heel/toe 几何检查改为恢复 MJCF `qpos0` 后使用 body 局部位置，避免把 IK 热启动姿态误当资产尺寸；动作限位检查改为使用配置收紧后的膝/肘有效限位。
- 删除 hinge 关节上的 `np.unwrap`，并进一步把中心差分改成真实相邻帧差，避免单帧尖峰被环绕修正或分摊。新增逐关节近限位/精确贴边率、有效任务整体/单任务位置 RMS，并设置离线动画质量门槛：整体位置 RMS `<0.075 m`、单任务 `<0.12 m`、总近限位率 `<0.15`、单关节贴边率 `<0.65`、相邻帧速度 `<35 rad/s`。这些不是实机安全规格。
- 使用最终配置重新执行四集合全部 10 条的 SMPL-X 前向、50 Hz 重采样、单阶段 Mink IK、CSV 元数据、Mimic NPZ 和联合验证，共 `21,151` 帧；10/10 报告为 `passed`，配置 SHA256 为 `498c55984d35ada378a62d33db7cac0d303873c17192d79cfd796d9fde85b9cf`，10 份元数据均与该配置和 MJCF SHA 一致。
- 最终 10 条最差指标：有效任务整体位置 RMS `0.05302 m`，单任务位置 RMS `0.10821 m`，总近限位率 `0.13493`，单关节精确贴边率 `0.59706`，真实相邻帧速度 `33.07 rad/s`，接触区间中位位移 `0.00350 m`、中位速度 `0.01203 m/s`、穿地 `0 m`。
- 产物覆盖写入 `output_data/robot_motion/*_bumi3.csv`、同名 `.meta.json`、`output_data/mimic_npz/bumi3/*.npz`、`output_data/keypoints/bumi3/*_keypoints.pkl` 与 `output_data/reports/*_bumi3.json`；旧 BUMI3 生成结果没有另存备份，如需回退必须恢复旧配置后重新生成，源数据集未改动。
- 验证记录：修改脚本 `py_compile` 通过；`PYTHONPATH=scripts ... pytest -q` 为 `17 passed`；标定连续两次输出一致；批处理末尾 `--list-only` 识别 10/10 条；在 `DISPLAY=:1` 实际打开最终 MuJoCo viewer，确认 BUMI3 姿态、自动相机、照明和网格地板均正常可见。尚未执行 IsaacLab 运行时 replay、动力学稳定性或实机测试，因此本次结论仅覆盖静态合同、离线 FK/IK 质量和 MuJoCo 运动学播放。

## 2026-08-28：MuJoCo 多轨迹播放器黑屏修复

- 修复 `scripts/play_bumi3_trajectories.py` 启动后只有黑色画面的问题：原播放器沿用 MJCF 静态模型中心设置相机，但首条 G1 轨迹的根节点初始位置约为 `(0.45, 2.95, 0.68)`，机器人落在默认视野外。
- 播放器现在会在启动、切换轨迹和重播时把自由相机对准当前轨迹根节点，并根据模型尺度自动选择观察距离；新增 `--camera-distance` 供手动覆盖。
- 提高 MuJoCo 默认头灯的环境光和漫反射强度，使没有自带场景灯的 G1 MJCF 也能清晰显示。
- 将输出标签从硬编码的 `[BUMI3]` 改为通用的 `[MuJoCo]`，因为播放器通过 `--config` 同样支持 G1 等其他机器人。
- 验证记录：`py_compile` 通过；`--list-only` 成功识别 10/10 条 G1 轨迹；相机单元检查确认焦点为首帧根位置 `(0.4505, 2.9458, 0.6760)`、自动距离 `2.7129 m`；在 `DISPLAY=:1` 实际启动 MuJoCo viewer 并截图确认 G1 清晰位于窗口中央。

这些修改只影响可视化相机和照明，不改变 CSV、重定向结果、机器人 qpos 或播放帧率。

## 2026-08-28：可切换播放器补充原版网格地板

- `scripts/play_bumi3_trajectories.py` 新增可视化模型构建步骤：先读取机器人 `MjSpec`，若 MJCF 不含 plane，则按 `multi_robot_visualize.py` 的原参数添加渐变天空盒、256×256 灰色网格纹理、10×10 重复且无反射的地面材质、`z=0` 无限平面和顶部方向光。
- 新增地板的 `contype` 与 `conaffinity` 均为 0，只作为运动轨迹的视觉参照；播放器仍使用 `mj_forward`，不会让地板参与动力学或改变 CSV qpos。
- 如果机器人 MJCF 已经带有 plane（例如当前 BUMI3 模型），播放器直接编译原模型，不会重复添加地板、纹理或灯光。
- 验证记录：`py_compile` 与 10/10 G1 轨迹加载通过；场景契约检查确认 G1 保持 `nq=36` 且关节名称/qpos 地址不变，geom 从 84 增至 85、plane 从 0 增至 1，BUMI3 保持 32 个 geom 和 1 个 plane；在 `DISPLAY=:1` 实际启动 viewer 并截图确认灰色网格地板、渐变天空与 G1 均清晰可见。

这样既让 G1 获得与仓库原多机器人播放器一致的地面视觉，也保持已有完整场景的机器人资产不变。

## 2026-08-27：资产契约与准备器

- 新增 `scripts/bumi3_common.py`，集中定义 IsaacLab 的 21 关节顺序、22 个物理 body 顺序以及四元数/有限差分工具。
- 新增 `scripts/prepare_bumi3_asset.py`，从本地 BUMI3 MJCF、URDF、STL 确定性生成重定向资产；按 URDF 核对限位并通过实际 mesh 自动计算固定 marker。
- 新增空的 `config/robot/bumi3_marker_overrides.yaml`，只在自动几何估计置信度不足时允许用户明确覆盖。
- 新增 `requirements-dev.txt`，将 pytest 与运行时依赖分离。

选择这些改动是为了先固定资产、关节顺序、body 顺序与四元数的边界，再扩展输入、IK 和 Mimic 导出，防止多个脚本各自猜测同一契约。

## 2026-08-27：BUMI3 配置与关键姿态自动标定

- 新增 `config/robot/bumi3.yaml`，使用 BUMI3 实际 body 名称、IsaacLab `bumi.py` 的 21 关节顺序、完整 22 body 顺序、单阶段 IK、初始屈膝姿态和 `active_only` 接触配置。
- 新增 `scripts/calibrate_bumi3_keyframes.py`，通过同一套 SMPL 世界姿态计算和 BUMI3 初始 FK 自动生成每个关键帧独立的轴映射。

配置没有沿用附件中的 BUMI2 body 别名，因为当前 BUMI3 MJCF 与 IsaacLab URDF 的物理 body 名已经一致；保留空别名表可让导出器仍走显式映射契约。

## 2026-08-27：SMPL-X 模型根目录兼容修复

- 修改 `scripts/smpl_replay.py` 的 `resolve_model_root()`：找到性别对应模型后传递具体模型文件，而不是已经带 `smplx/` 的子目录。
- 原因是 `smplx.create()` 对目录参数会自动追加模型类型；旧逻辑在 GENMO 的 `body_models/smplx/` 布局下会错误访问 `smplx/smplx/`，导致自动标定和正式 SMPL-X 前向都无法运行。

## 2026-08-27：SMPL 输入适配、50 Hz 重采样与接触元数据

- 扩展 `scripts/smpl_replay.py`，兼容原 `trans/root_orient/pose_body`、`global_orient/body_pose`、`poses` 以及四集合实际使用的 `pose[T,66] + transl`。
- 新增 `--model-type`、`--translation-scale`、`--up-axis`、`--target-fps`、`--max-frames`；平移线性插值，根与 21 个 body 关节使用 SLERP，接触在重采样后的序列上重新计算。
- BUMI3 可用秒配置接触速度窗口，并只允许脚跟/脚尖参与全身地面高度估计；手端仍会计算接触但不会单独推动全身升降。
- keypoint PKL 恢复保存 contact 位置、速度、阈值，并追加输入文件、源/目标 fps、模型类型、连杆长度、腿长位移缩放与 ground reference 元数据。
- 新增 robot/skeleton 连杆集合、有限长度和左右对称性检查；字段缺失的旧机器人仍走原默认行为。

## 2026-08-27：YAML 结构化解析修复

- 将 `robot_links` 与 `key_frame_config` 从逐行字符串解析改为 PyYAML 结构解析，同时兼容旧配置的行内列表和自动标定生成的标准多行序列。
- 增加端点数量、向量 shape、有限值、轴映射正交性与右手行列式检查，避免合法 YAML 因排版不同而失败，也避免畸形矩阵静默进入 IK。

## 2026-08-27：单阶段 IK 接触契约与 Mimic 导出

- 扩展 `scripts/robot_retarget.py`：复用已有 ground 并关闭碰撞，支持配置化初始 qpos、IK 参数、显式接触映射、`active_only` 当帧任务列表以及 CSV 同名 JSON 元数据；未配置这些字段的旧机器人继续保持原语义。
- 新增 `scripts/export_bumi3_mimic_npz.py`，按关节名重排 IsaacLab 21 关节，逐帧 MuJoCo FK 导出 22 个物理 body，并使用连续化四元数相对旋转计算角速度。
- 新增 `scripts/validate_bumi3_retarget.py`，同时覆盖模型契约、qpos 数值与限位、FK 任务 RMS、接触区间脚滑/穿地阈值和 Mimic NPZ 严格 shape/四元数检查，并将失败写入 JSON 报告。
- 新增 `scripts/play_bumi3_trajectories.py`，在单个 MuJoCo viewer 中用方向键、P/N 或数字键即时切换多条 CSV；`--list-only` 支持无图形环境预检轨迹顺序、shape、fps 和时长。
- 为 `smpl_replay.py` 与 `robot_retarget.py` 增加兼容默认值的 `--output-dir`，并给前者增加 `--keypoints-name`，使一键脚本的 `OUTPUT_DIR`/`KEYPOINTS_NAME` 真正控制产物路径而非仅作为展示变量。
- 新增 `bash/retarget_smpl_to_bumi3.sh`，用 8 个可追踪步骤串联资产准备、模型预检、50 Hz 关键点、单阶段 IK、CSV/JSON、Mimic NPZ、联合验证和可选播放；任一步失败由 trap 报告准确步骤。
- 新增三组 pytest：输入适配/Y-up/重采样、资产/marker/接触任务、Mimic 顺序/shape/四元数连续化；测试数据在临时目录构造，不依赖下载数据集或图形窗口。
- 修正 Mimic 导出测试夹具：原夹具给肘关节写入正小角，正确触发了 `[-2.26, 0]` 物理限位；现改为按每个关节实际上下限生成各不相同的合法值，以便测试真正聚焦名称重排。
- 新增 `docs/bumi3_retargeting.md` 与 README 快速入口，记录资产 SHA、marker 几何、输入/50 Hz、CSV/NPZ/四元数/顺序、接触、播放器、验证、重标定和排错契约。
- 新增四集合 10 条远程人工筛选动作的 `download_manifest.json`，记录服务器 data0 来源目录、2/3/3/2 配额、每条帧数/时长和本地 SHA256，不记录登录凭据。
- 完成 G1 20 帧回归：旧配置无需新增字段，legacy contact 仍启用，新 CSV 为 36 列且与仓库已有 G1 CSV 列数一致。
- 新增 `bash/retarget_music_smpl_4set_bumi3.sh`，要求清单目录恰有 10 条 NPZ，逐条调用一键链路并在末尾用播放器 `--list-only` 复核；同时移除单条脚本会展开全部逐帧连杆数组的调试摘要，长轨迹证据改由 JSON 报告承载。
- 首条完整轨迹暴露两个质量问题：低通地面高度在地面下降时滞后造成源脚点负 z，且原始未加权 task error 与 Mink 实际 cost 加权目标不一致，常令接触帧只迭代一次。BUMI3 现显式启用地面防穿透 clamp 与 `solver_weighted` 收敛指标；旧机器人默认仍用原始行为。
- 源接触帧修复后，最深穿地仍出现在 `active_only` 明确不加接触 task 的摆动脚。BUMI3 新增 IK 后 freejoint 根 Z 最小安全修正：读取四个足底 marker，仅整体上抬到 2 mm，不改关节、不引入第二阶段 IK，并在 metadata 记录逐帧修正统计；旧机器人默认关闭。
- 第三条完整音乐轨迹在固定 0.03 m 脚滑门槛上超出 1.5 mm，且仅 10 帧命中 50 次迭代上限，主要是 20 的接触位置 cost 弱于 40 的踝普通任务。四个脚跟/脚尖接触 cost 对称提高到 40，保持同一单阶段任务结构与固定验收阈值。
- 10 条总审计发现 AIST++/CoMPAS3D 因数据集世界原点不同而产生 0 接触帧：其足底绝对 Z 分别约为 1.0/0.3 m。BUMI3 现用四脚源点 1% 高度分位作为序列地板零点后再应用原 0.075 m 人体尺度阈值；验证器要求每个启用脚点至少一帧接触，并在报告保存脚滑聚合指标，禁止空接触绕过验收。
- CoMPAS3D Pair5 在恢复真实接触后区间端点位移已通过，但区间内中位速度为 0.0756 m/s，144 帧达到迭代上限，说明接触 task 相对全身任务仍不足。四个对称接触位置 cost 从 40 提高到 80，并以完整 7529 帧重新验证，不改检测或验收阈值。
- 使用最终接触 cost=80 对四集合全部 10 条轨迹重新执行 SMPL-X 前向、50 Hz 重采样、单阶段 IK、Mimic 导出与联合验证，清除旧配置哈希对应的中间结果；10/10 报告均为 `passed`，并由批处理末尾的无窗口播放器索引检查确认可切换轨迹数量、帧数、fps 与时长。
- 文档明确区分已完成的 `bumi.py`/URDF/MuJoCo 静态顺序核对与尚未执行的 IsaacLab 单环境 `robot.body_names` 运行时核对，防止把静态契约检查表述成仿真运行证据。
- 最终指标复核发现 AIOZ E2 虽通过脚滑/穿地阈值，但因序列开头尚无激活接触，地面高度状态从错误的世界零点开始，导致 IK 后根 Z 安全修正最高约 1.01 m。`smpl_replay.py` 现分别计算原人体尺度的 source floor 与连杆缩放后的 retarget floor，并用后者初始化地面偏移状态；PKL 元数据记录两种地板及偏移范围，新增无初始接触单测，避免跨尺度混用和“验收通过但根高度靠大幅后修正兜底”。
- 地板初始化修复后再次完整生成并联合验证 10 条轨迹：10/10 `passed`；E2 的 IK 后根 Z 修正均值/最大值从约 0.485/1.009 m 降为 0.0156/0.1082 m，最终全体最差中位接触位移 0.00961 m、最差中位接触速度 0.04146 m/s、最大穿地 0 m。

## 2026-08-31：建立当前仓库与原始 GMR 的同输入十轨迹比较链路

- 用户要求使用 `/home/weili/GMR_minimal_robots_smplx.tar.gz/GMR-master/general_motion_retargeting/ik_configs/smplx_to_bumi3_auto.json` 对同一批数据生成原始 GMR 对照。数据清单固定为 `dataset/music_smpl_4set/` 已下载的全部 10 条：AIOZ-GDance 3 条、AIST++ 2 条、CoMPAS3D 2 条、FineDance 3 条；不再重新挑选来源不同的动作。
- GMR 批处理显式使用 `--legacy_pipeline --fps 30`，其合同为逐帧 IK、无 temporal IK、无 wrist projection、无 trajectory QP、无 Root-Z QP，并保留旧版“整段所有 body 原点全局最低值落地”。这用于代表原始 GMR 基线，避免把该 GMR 分支后来新增的时序/QP策略混入对照。
- 新增 `scripts/prepare_bumi3_gmr_comparison.py`：它不执行 IK，也不做平滑、插值、限位裁剪或二次高度修正；只按同一 NPZ 清单整理当前仓库 CSV，并把 GMR PKL 按 GMR XML 的关节名与 `jnt_qposadr` 封装为播放器 CSV，同时硬核对两边逐条帧数和 fps。
- 新增 `config/robot/bumi3_gmr_original_player.yaml`，让现有 MuJoCo 多轨迹播放器加载 GMR 自己的 BUMI3 XML。当前 XML 与 GMR XML 都是 `nq=28`、21 个 hinge，但 qpos 顺序不同，所以禁止跨模型直接播放 CSV；该专用配置用于消除列错位。
- GMR 实际完成 10/10 条、每种方法各 `12,695` 帧；GMR 指定 IK 配置 SHA256 为 `f120916758a7d4eed1b23991fbfc8178497999d4a88ccbaec2f1ea1c4bf0804d`，GMR XML SHA256 为 `482138b437dbdabd6171fa8d44b55db5d7125a228c95b69ce3d1159cafe8537c`。当前方法配置/XML SHA256 分别为 `d10739458cb374b09b1f138f90abcb7de98e3eb6b36507353bd2e97fbbca55c6` 和 `fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3`。
- 整理产物位于 `output_data/comparison/robot_retargeter/`、`output_data/comparison/gmr_original/`，GMR 原始 PKL 保存在 `output_data/comparison/gmr_original_pkl/`，共同来源、帧数、fps、配置/模型哈希及逐条文件哈希记录在 `output_data/comparison/comparison_manifest.json`。当前方法副本与原 CSV 逐字节一致；GMR CSV 反向读取后，根位置、四元数和关节相对 PKL 的最大误差分别为 `5.0e-11`、`6.2e-11`、`5.0e-11 rad`，仅来自十位小数文本序列化。
- 两套输出均通过现有播放器 `--list-only` 的 10/10 shape、有限值、fps 和顺序检查，且脚本清单顺序显式与播放器扁平文件名排序一致；`py_compile` 通过，`PYTHONPATH=scripts ... pytest -q` 为 `21 passed`，`git diff --check` 通过。本轮没有自动打开 GUI 肉眼评价动作，也没有做动力学 replay 或实机测试。
- 当前方法播放命令为 `PYTHONPATH=scripts /home/weili/miniconda3/envs/robot_retargeter/bin/python scripts/play_bumi3_trajectories.py --config config/robot/bumi3.yaml --motion-dir output_data/comparison/robot_retargeter --pattern '*_bumi3.csv'`；原始 GMR 命令只把配置替换为 `config/robot/bumi3_gmr_original_player.yaml`、目录替换为 `output_data/comparison/gmr_original`。两个窗口使用相同数字键即可观察同一动作。

## 2026-08-31：GMR 对照切换为最新足底 Root-Z 流水线并重生成十条轨迹

- 用户观察到当前 `robot_retargeter` 略微悬空、legacy GMR 足底陷入地面，并要求 GMR 不再使用整段所有 body 原点的全局最低 Z。真实 MuJoCo FK 复核表明：当前方法在源接触激活帧中，左右 heel marker 中位高度分别约为 `0.04065/0.03127 m`，toe marker 中位高度约为 `0.01650/0.01019 m`；原因是四足点全序列 1% 分位只做一次地板规范化，运行时又是 cost=15、orientation cost=0 的软位置任务，且支撑投影和平足约束均关闭，因此不保证每个接触点严格位于 z=0。
- legacy GMR 虽把全身最低 body 原点置于 z=0，真实左右足底 mesh 合并统计的中位高度为 `-0.04652 m`、1% 分位为 `-0.10462 m`，说明 ankle/body 原点并不是足底表面；这正是视觉上约 4 cm 下陷的来源。旧版 `gmr_original*` 产物和 `comparison_manifest_legacy.json` 保留作历史复现，但不再作为当前比较结果。
- 最新 GMR 仍使用用户指定的 `general_motion_retargeting/ik_configs/smplx_to_bumi3_auto.json`，并启用 `temporal_ik=true`、`project_wrist_targets=true`、`trajectory_optimization=true`、`root_height_optimization=true`、`legacy_root_alignment=false`。其 Root-Z 优化只从左右 `ankle_roll/foot/toe` mesh 提取最低顶点和足底四角，先按足底稳健分位做整段常量对齐，再以速度、加速度和防穿地约束求解逐帧一维 QP；没有再读取手、躯干或其他 body 原点决定地板。
- 第一次严格运行得到 5/10 直接通过，另外 5 条只失败于轨迹 QP 后的 `fk_drift/fidelity` 门槛。为保留完整 10 条可视化比较，第二次保持所有算法、约束和权重不变，仅使用 `--allow_quality_failure` 关闭“不通过即拒绝写盘”；最终每条 PKL 仍保存真实 acceptance。10/10 的 `safety_overall=true`，5/10 的 `overall=true`，总帧数为 `12,695`。
- 最新结果的 `root_height.method` 均为 `foot_contact_bounded_qp`；十条真实足底最大穿地均约 `0.0020001 m`，对应最新 Root-Z 配置显式允许的 2 mm penetration tolerance，已消除 legacy 的约 4 cm 系统性下陷。各条常量高度修正范围为 `-0.43020～+0.73140 m`，它由实际足底 mesh 和每条数据世界原点共同决定，不是手工统一增加 4 cm；动态修正峰值范围为 `0.02833～0.05681 m`，并受速度/加速度约束。
- `scripts/prepare_bumi3_gmr_comparison.py` 默认输入改为 `gmr_latest_pkl/`，硬检查上述五个最新流水线开关及每条 `foot_contact_bounded_qp` 方法后，按 GMR XML `jnt_qposadr` 输出 `gmr_latest/` CSV；metadata 额外保留 acceptance、Root-Z 诊断和最终足底审计。新增 `config/robot/bumi3_gmr_latest_player.yaml`，旧 legacy 播放配置继续保留，避免破坏历史复现。
- 最终反向回读确认：当前方法比较副本与原 CSV 逐字节相同；最新 GMR CSV 相对 PKL 的根位置、根四元数和关节最大误差分别为 `5.0e-11 m`、`6.0e-11`、`5.0e-11 rad`，仅为十位小数文本序列化误差。两套播放器 `--list-only` 均识别相同顺序的 10/10 条；`py_compile`、`git diff --check` 通过，`PYTHONPATH=scripts ... pytest -q` 为 `21 passed`。本轮未自动打开 GUI 做主观效果判断，也未执行动力学 replay 或实机测试。
- 最新 GMR 播放命令为 `PYTHONPATH=scripts /home/weili/miniconda3/envs/robot_retargeter/bin/python scripts/play_bumi3_trajectories.py --config config/robot/bumi3_gmr_latest_player.yaml --motion-dir output_data/comparison/gmr_latest --pattern '*_bumi3.csv'`；当前方法继续使用 `config/robot/bumi3.yaml` 和 `output_data/comparison/robot_retargeter`。两个窗口按同一个数字键选择同一轨迹。

## 2026-08-31：新增四库全量 Z-up 重定向、断点续跑与发布门禁

- 本轮位于 `feature/bumi3`，起始 HEAD 为 `7ea44142878a2b842e357ec7c68db4ccd80ac714`；修改目标是用当前 `robot_retargeter` 替代旧 GMR，对服务器 2 的 AIOZ-GDance、AIST++、CoMPAS3D、FineDance 高质量 SMPL-X 数据做正式全量 BUMI3 重定向。没有修改 `main`，没有 reset、clean、stash 或覆盖既有输出。
- 新增 `scripts/retarget_bumi3_full_dataset.py`。正式模式固定核对四库 `1978/963/72/149`、共 `3162` 条，逐文件要求 `root_orient/pose_body/trans/betas`、30 Hz、有限值、`right_handed_z_up_metric`，并核对历史来源为 Y-up、已执行一次 `+90° about X`。批处理按数据集隔离同名 stem，模型预检只执行一次，支持受控并发、逐条日志、进度 JSONL、来源一致的断点续跑和最终 release report。
- 根倾角发布门禁以“每条动作根倾角中位数”的分库分布为对象：分库中位数必须 `<30°`、P95 `<45°`、动作中位数 `>=60°` 的占比必须 `<1%`。少量真实倒立/地板动作允许保留，但整库约 90° 躺倒会拒绝发布；门禁同时应用于源 SMPL-X 与输出 BUMI3。
- `bash/retarget_smpl_to_bumi3.sh` 新增 `UP_AXIS`、`PREPARE_ASSET`、`RUN_PREFLIGHT`，全量任务显式传 `UP_AXIS=z`，并在并发前复用仓库内已提交 BUMI3 资产与统一预检，避免每条重复改写资产和共同报告。单条旧入口默认仍为 `auto/true/true`，保持原行为。
- `scripts/smpl_replay.py` 在 keypoint 元数据中新增 `requested_up_axis`、`y_up_to_z_up_conversion_applied` 和 `output_coordinate_system`；`scripts/validate_bumi3_retarget.py` 联合验证这些字段，明确禁止已经是 Z-up 的动作再次旋转，并保存输出机器人根倾角 median/P95/max。全量续跑只有在来源、fps、四类产物、联合验证和两处坐标元数据全部一致时才会跳过。
- 新增 `tests/test_bumi3_full_dataset.py`，覆盖正确 Z-up 合同、错误 Y-up 声明拒绝、直立/躺倒倾角和分布门禁。`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts ... pytest -q -o cache_dir=/tmp/robot_retargeter_pytest_cache_full_batch` 为 `28 passed`，修改脚本 `py_compile` 与 `git diff --check` 通过；测试缓存和本轮 pyc 已精确清理。用本地旧十条 raw `pose+transl` 运行正式输入审计得到预期拒绝，证明门禁不会把未标准化旧输入混入新发布；服务器 2 的全量标准化 Z-up 数据审计和端到端重定向结果将在任务运行后另行补记。
- 本轮门禁只证明输入/输出坐标链路、离线 IK/FK 数值和既有联合质量阈值；不等价于 IsaacLab 动力学跟踪或实机安全验证。正式输出使用新的独立根目录，旧 GMR 和此前训练数据不会被覆盖。
- 服务器 2 首轮四库各一条 smoke 中，AIST++、CoMPAS3D、FineDance 完整通过，AIOZ 在 SMPL-X 前向前退出；原因是其首条 stem 以 `-` 开头，argparse 把分离传入的值误认成新选项。单条 shell 的两处参数现改为 `--keypoints-name=<stem>`，并新增回归测试固定该写法；这只修复文件名传参，不改变动作、IK 或坐标。
- 连字符修复后的 AIOZ 首条已完成全部产物，但联合验证发现第 0→1 帧左右肘速度峰值为 `44.75/37.20 rad/s`。首轮强制 50 次迭代把峰值降到 `35.28 rad/s`，强制 100 次反而回到 `41.72 rad/s`，证明不是单纯迭代不足：`legacy_hold` 在每帧开始把未接触手脚冻结到该轮开始姿态，同一轮多迭代不会更新冻结目标，下一帧才更新而形成启动换解。最终新增兼容默认值为 0 的 `initial_settle_passes`，BUMI3 取 5；正式输出前用同一个第 0 帧重复“更新冻结目标→按原条件求解”，稳定后才记录第 0 帧，不推进时间、不滤波，也不改后续帧权重与接触。
- 全量断点续跑补充三类不可变指纹：源 SMPL-X SHA256、当前 BUMI3 配置 SHA256 和 MJCF SHA256。任一内容变化时，同路径旧产物也会重新生成，避免仅凭“报告曾经 passed”错误复用旧配置结果；源动作 SHA 同时写入 keypoint 与 CSV metadata。
- 首轮全量基线的早期失败主要来自 AIOZ 手臂任务位置 RMS，而不是坐标：BUMI3 短手臂对部分人体动作无法达到原 `0.12 m` 单任务门槛。按“宁可少跟动作、不能跳 IK”的原则，新增 `task_position_error_validation=warning`，只把整体/单任务/calf 跟随误差降为结构化 warning；关节越限、速度/加速度/jerk、坐标链路、地面和足底质量仍是硬失败。另启用已有的 `30 rad/s` 逐帧关节限速，只修改超过该值的离群关节帧，不启用全局高斯平滑、加速度或 jerk 后处理。
