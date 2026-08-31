# BUMI3 人体动作重定向与 IsaacLab Mimic 导出

本文说明本仓库 BUMI3 一等支持的资产来源、数据契约、单阶段 IK 方法、运行命令、
验证边界与排错方式。实现保持仓库原有 SMPL/SMPL-X FK → 按机器人连杆缩放 →
接触/地面修正 → 单阶段 Mink IK 主干，没有引入 GMR、两阶段 IK、轨迹 QP 或
学习式后处理。CSV 保持原格式，额外的 JSON 与 Mimic NPZ 用于明确四元数、顺序、
资产 SHA 和质量统计。

## 1. 资产来源与派生边界

当前派生来源为：

```text
../legged_lab/source/NoetixRobot/NoetixRobot/assets/robots/bumi3
├── mjcf/bumi3.xml
├── urdf/bumi.urdf
└── meshes/*.STL
```

- 源 MJCF SHA256：`041c81e8176c7f375302796deca28b141891a3c097d8e341e8d967b735466edf`
- 源 URDF SHA256：`174c1747019ced64267e74244bf89f3746856c90c30f88e4f162582ebc486476`
- 当前派生 MJCF SHA256：`fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3`

`prepare_bumi3_asset.py` 保留源 MJCF 的 body 层级、质量、惯量、关节轴、执行器、
碰撞和传感器；只允许按 URDF 核对/修正位置限位、修正相对 `meshdir`、加入 marker
以及保留一个可视化 ground。当前 URDF/MJCF 的 21 个位置限位完全一致，因此报告中
`joint_limit_changes` 为空。每次派生结果和网格 SHA 记录在
`asset/robot/bumi3/prepare_report.json`。

GMR 仓库的 BUMI3 XML 只作为 marker/重定向结构参考；它与当前 legged_lab 资产的
动力学参数和 SHA 不一致，因此没有替换这里的质量、惯量、执行器或限位。

## 2. 为什么使用 fixed body marker

Mink 的 `FrameTask(frame_type="body")`、接触映射、MuJoCo FK 校验和调试可视化都使用
统一的 body 名称接口。固定 body 能成为明确的坐标系和四元数目标，而 site 只提供
点/局部朝向且会迫使不同脚本维护两套 frame type。marker body 没有关节、惯性，
其透明 sphere geom 使用 `density=0 contype=0 conaffinity=0`，所以不增加质量、自由度
或碰撞。

九个 marker 为：`hips_sphere`、`neck_sphere`、`head_sphere`、左右
`foot_end_link`、左右 `toe_link`、`left_hand`、`right_hand`。

## 3. marker 计算方法

- 髋中心：左右 `leg_pitch` 零位姿关节锚点的世界坐标均值，再转回 `base_link` 局部坐标。
- 肩/颈中心：左右 `arm_pitch` 零位姿关节锚点均值，再转回 `waist_yaw_link` 局部坐标。
- 头：在 waist mesh 顶端基础上加一个由 mesh 尺寸约束的小余量，位于肩中心上方。
- 脚跟/脚尖：读取左右脚 STL，在最低 1% 附近建立 4 mm 足底带，沿其主要前后 X 轴取
  2%/98% 分位并略向内收；同时要求 X 跨度显著大于 Y 跨度、toe 在 heel 前方。
- 手端：读取前臂 STL，以 PCA 第一主轴两端 2% 顶点的中位点选择离肘关节更远的一端。

自动估计若缺乏几何置信度会报错。确需人工覆盖时只修改
`config/robot/bumi3_marker_overrides.yaml` 并重新运行准备器，不要直接手改生成 XML。

## 4. SMPL/SMPL-X 输入格式

支持以下核心字段组合：

```text
trans + root_orient[T,3] + pose_body[T,63]
trans/transl/translations + global_orient[T,3] + body_pose[T,63]
trans/transl/translations + poses[T,D>=66]
trans/transl/translations + pose[T,D>=66]
```

平移字段优先级为 `trans` → `transl` → `translations`；帧率优先读取
`mocap_frame_rate`、`mocap_framerate`、`fps`、`frame_rate`、`framerate`，缺失时明确
默认为 30 Hz。模型类型可从 `surface_model_type` 读取或用 `--model-type` 指定；缺失
元数据的本次四集合标准数据必须指定 `MODEL_TYPE=smplx`。`betas` 缺失时为 10 维零，
gender 缺失时为 neutral。`--translation-scale` 只能是正有限值，不做米/毫米猜测。

本次四集合数据声明 `right_handed_y_up_metric`，流水线通过固定世界 X 轴 +90° 同时
旋转根平移和根朝向，转为 MuJoCo Z-up。

## 5. 帧率与 50 Hz 的来源

50 Hz 不是 BUMI3 几何或 Mink IK 的固有要求。它来自当前 legged_lab BUMI3 IsaacLab
环境的 `sim.dt=0.005` 和 `decimation=4`：策略/动作每 0.02 秒更新一次，因此部署边界为
50 Hz。离线重定向可以使用其他真实时间采样率；当前严格 G1 算法基线默认 30 Hz，和
仓库现有 G1 对照轨迹一致，以排除帧率与算法策略同时变化的干扰。

输入动作跨帧率转换不能只改 metadata：平移与表情按真实时间线性插值，根旋转、21 个
body rotvec、手/颌/眼旋转都先转 quaternion 后用 SLERP，再转回 rotvec。速度/高度
接触检测在重采样后的序列上重新计算，输出 PKL、CSV metadata 和 NPZ 的 `fps` 均记录
实际采样率。当前 30 Hz 基线只用于 MuJoCo 运动学对照，不能未经显式重采样就冒充
50 Hz 控制器输入。

## 6. 单阶段 IK 与接触

`config/robot/bumi3.yaml` 仅启用一次 Mink IK；每帧以上一帧 qpos 为 seed。首帧从
`model.qpos0` 开始，按名称写入初始根 `wxyz` 和 21 个关节，并在膝/肘收紧限位后
检查合法性。`max_ik_iterations`、误差改善阈值、solver 和 damping 全部来自 YAML。

人体脚跟、脚尖和手端仍计算速度+高度接触状态，但 BUMI3 生产配置沿用仓库 G1 的
`contact_task_mode: legacy_hold` 和 `contact_pos_fixed_factor: 15` 低权重接触任务。
接触区间内使用源足点/手点均值作为目标；摆动期任务仍驻留，但目标被更新为机器人
当前接触点，所以瞬时误差为零。这样 QP 任务维度不因接触边界变化，也不会出现高权重
heel/toe 锁点突然接管全身 IK 的分支跳变。

地面高度参考被明确限制为 `left_foot_end`、`left_toe`、`right_foot_end`、
`right_toe` 四个足点；左右手可以继续作为低权重 IK 接触任务，但绝不参与地面估计。
生产配置恢复仓库默认的 `contact_height_dynamic_offset_enabled: true`，但动态高度函数只
接收上述四个足点的状态与位置；即使手部检测到接触，也没有进入整机 Z 偏移的路径。
不同数据集的世界原点并不一致，因此仍设置
`contact_height_relative_to_sequence_floor: true`，但不再取全序列四足点高度的 1%
分位。新方法先筛选速度不超过 `0.2 m/s` 的
稳定候选，再在高度轴寻找宽度不超过 `0.08 m` 的最大密集簇；同样大小时选更低的簇，
最后以候选中心 ±`0.04 m` 内点的中位数拟合一条恒定水平地板。MuJoCo 播放器地面是
水平 `z=0`，所以当前选择常数模型而非斜平面；拟合样本数、内点数、MAD 与最终高度
都会写入 PKL。该值只初始化高度滤波，后续逐帧偏移仍只随四足稳定接触变化。

为改善平足但不恢复高权重锁点，左右脚各增加一个标量软任务
`heel_z - toe_z = 0`。它只在同一只脚 heel/toe 同时接触时启用，代价为 `15`（仍低于
腿部位置任务的 `30`），以 `0.10 s` smootherstep 渐入渐出；不锁 XY、不规定绝对
地板高度，也不约束脚掌完整
朝向。其雅可比是两个足点世界 Z 雅可比之差，统一 Root-Z 平移严格相消，因此不会
直接上下搬动整台机器人。toe-off 或摆动期允许 heel/toe 高度不同。

生产配置仍关闭接触滞回、接触前竖直预约束、成对绝对锁点/平足方向任务、IK 后根 Z
支撑投影和足底屏障。代码保留这些通用能力供其他机器人或显式实验配置使用。当前
软任务只改善 heel/toe 相对高度，不承诺 `legacy_hold` 的每个支撑帧严格无滑移、无
局部穿透；这些问题仍需结合红色 heel/toe 点和验证报告检查。

严格基线还把 hips/head/hip/thigh/calf/shoulder/arm/forearm 的位置与旋转 cost 逐项
设为 G1 配置值，使用 G1 的 `legacy_raw` 停止误差，不启用时间姿态正则、输出关节
速度/加速度/jerk 修正或高斯平滑。BUMI3 自己的 MJCF、body 名称、连杆尺度、方向标定、
IK 热启动姿态和真实关节限位必须保留；这些是机构合同，不是可消除的算法变量。

## 7. 一键运行

先激活指定环境：

```bash
conda activate robot_retargeter
```

然后运行：

```bash
BUMI_SOURCE_DIR=../legged_lab/source/NoetixRobot/NoetixRobot/assets/robots/bumi3 \
SMPL_MOTION_FILE=dataset/music_smpl_4set/aistpp/gBR_sBM_cAll_d04_mBR0_ch01.npz \
SMPL_MODEL_PATH=../GENMO/inputs/checkpoints/body_models \
MODEL_TYPE=smplx \
TARGET_FPS=30 \
VISUALIZE=false \
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
./bash/retarget_smpl_to_bumi3.sh
```

可用变量：`BUMI_SOURCE_DIR`、`SMPL_MOTION_FILE`、`SMPL_MODEL_PATH`、`MODEL_TYPE`、
`TARGET_FPS`、`KEYPOINTS_NAME`、`OUTPUT_DIR`、`RENDER_DEBUG`、`VISUALIZE`、
`MAX_FRAMES`、`PYTHON_BIN`。`MAX_FRAMES=0` 表示完整动作。

## 8. CSV 与 JSON 元数据

CSV shape 为 `[T,28]`：

```text
root_pos_xyz[3] + root_quat_xyzw[4] + MuJoCo 标量关节 qpos[21]
```

CSV 延续仓库已有的 `xyzw` 根四元数约定。每个 CSV 有同名 `.meta.json`，记录源动作、
实际 fps、帧数、qpos 大小、MuJoCo 关节名和 qpos 地址、Isaac 顺序、XML/config SHA、
接触激活帧/区间数以及逐帧最终 IK 误差和迭代次数。导出器会核对元数据与当前 XML
及当前配置，拒绝模型或配置漂移后继续静默导出。

## 9. Mimic NPZ

输出键和 shape：

```text
fps                 scalar, 当前基线为 30.0
joint_pos           [T,21]
joint_vel           [T,21]
body_pos_w          [T,22,3]
body_quat_w         [T,22,4], wxyz
body_lin_vel_w      [T,22,3]
body_ang_vel_w      [T,22,3]
```

另含 `joint_names`、`body_names`、`anchor_body_name=waist_yaw_link`、`source_motion`、
`robot_name=bumi3`、`quaternion_order=wxyz`。关节/线速度使用首尾单边、内部中心差分；
角速度在逐帧统一四元数半球后，以相对旋转 log/rotvec 除以对应时间计算，绝不直接差分
四元数四个分量。

## 10. MuJoCo 与 IsaacLab 名称和顺序

当前 BUMI3 物理 body 名在 MJCF 与 IsaacLab URDF 中一致，
`mjcf_to_isaac_body_name` 因而为空；导出器仍支持正向或反向 alias，方便未来资产改名。
虚拟 marker 不进入 22 body Mimic 数组。

Isaac 21 关节顺序不是 MuJoCo XML/qpos 原生顺序。权威列表在 `bumi3_common.py`，导出时
逐个以 `mj_name2id → jnt_qposadr` 取值，不能对 CSV 后 21 列直接假设重排。22 body
顺序同理由 `bumi.py`/配置固定，以名称逐个 FK 提取。

本次已把 21 关节和 22 body 名称/顺序与当前 legged_lab 的 `bumi.py`、URDF 层级和
MuJoCo 派生模型逐项静态核对；没有在本机启动 IsaacLab 单环境读取运行时
`robot.body_names`，因此这项仍属于 IsaacLab 运行时待验证边界，而不是已完成证据。

内部与 NPZ body 四元数为 `wxyz`；CSV 根四元数为 `xyzw`。唯一转换发生在 CSV
写入/重建边界，代码和元数据均显式标注。

## 11. 验证

模型预检：

```bash
python scripts/validate_bumi3_retarget.py \
  --config config/robot/bumi3.yaml \
  --report output_data/reports/bumi3_model_preflight.json
```

完整联合验证：

```bash
python scripts/validate_bumi3_retarget.py \
  --config config/robot/bumi3.yaml \
  --keypoints output_data/keypoints/bumi3/MOTION_keypoints.pkl \
  --csv output_data/robot_motion/MOTION_bumi3.csv \
  --metadata output_data/robot_motion/MOTION_bumi3.meta.json \
  --npz output_data/mimic_npz/bumi3/MOTION.npz \
  --report output_data/reports/MOTION_bumi3.json
```

验证器检查模型/marker/ground/对称性、CSV 有限值与实际越限、逐任务 FK RMS、最大
速度/加速度/jerk、根四元数、接触区间水平位移/速度/穿地，以及 NPZ 全部 shape 和
单位四元数。它还核对 keypoint 产物记录的 ground reference、动态高度策略和恒定高度
标定，防止旧 PKL 与新配置混用；生产静态地板模式要求四足点 1% FK 高度位于地面
±`0.02 m`，且根节点最低高度大于 `0.15 m`。速度、加速度和 jerk 在所有接触模式下
始终硬验收。只有明确启用
`paired_support` 时，脚滑、穿地、支撑点离地和平足 heel/toe 高度差才是硬门槛；当前
`legacy_hold` 配置把超出相同参考阈值的足部指标写为 warning。BUMI3 踝 roll 的物理
范围仅为 ±0.17 rad，连续贴限位不等于帧间跳变，因此生产配置也把近限位/精确贴限位
占比降为 warning，但任何真实越限仍会失败。

## 12. 多轨迹 MuJoCo 播放

完成多条轨迹后：

```bash
python scripts/play_bumi3_trajectories.py \
  --motion-dir output_data/robot_motion \
  --pattern '*_bumi3.csv'
```

控制：Space 暂停/继续；左键/P 上一条；右键/N 下一条；R 重播；1～9 与 0 直选第
1～10 条。无显示环境可加 `--list-only` 做发现/shape/fps/时长检查。

## 13. 重新标定 key_frame_config

当 BUMI3 初始姿态、body frame 或 SMPL 模型资产发生有意变化时运行：

```bash
python scripts/calibrate_bumi3_keyframes.py \
  --robot-config config/robot/bumi3.yaml \
  --smpl-model-path ../GENMO/inputs/checkpoints/body_models \
  --model-type smplx \
  --gender neutral \
  --write
```

脚本以同一套中性 SMPL-X 世界姿态和 BUMI3 初始 FK 计算每个 source/body 独立的
`R_source.T @ R_robot`，并验证正交性、右手行列式和中性残差。标定后必须重新运行
pytest、模型预检和至少 100～300 帧集成测试。

## 14. 常见错误

- `SMPLX_NEUTRAL.npz not found`：`SMPL_MODEL_PATH` 应指向包含 `smplx/` 的模型根或具体文件；
  四集合数据使用 `MODEL_TYPE=smplx`。
- 模型出现 `smplx/smplx`：确认使用本分支修复后的 `resolve_model_root()`。
- CSV 是 28 列但 NPZ 顺序错误：不要切片假设 MuJoCo 顺序，必须用导出器按名称取地址。
- 机器人侧躺或轴镜像：确认输入 `coordinate_system`/`--up-axis`，然后重新标定轴映射。
- 脚滑验证失败：先检查 contact 激活帧/区间、PKL 的 contact positions/speeds 和 IK
  误差；不要通过删除接触帧规避阈值。
- XML SHA 漂移：重新运行资产准备器、审查 `prepare_report.json` 并重新生成 CSV；不要
  用旧元数据配新资产。
- 远程数据来源和本地 SHA：见 `dataset/music_smpl_4set/download_manifest.json`。
