"""整段轨迹 QP 与手臂上一帧姿态正则的回归测试。

本文件使用仓库交付的真实 BUMI3 MuJoCo 模型构造包含单帧关节尖峰的轨迹，验证
优化结果不会修改 freejoint 根位姿，并且所有标量关节同时满足带安全裕度的位置
限位、按名称覆盖的速度上限、加速度上限和 jerk 上限。测试还核对上一帧姿态代价
只能落到配置指定的八个手臂自由度，腰腿和 freejoint 保持零权重，防止以后一个
看似无害的标量默认值把全身都锁到上一帧。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from trajectory_qp import (
    optimize_scalar_joint_trajectory,
    resolve_posture_cost_vector,
    scalar_joint_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config/robot/bumi3.yaml"
XML_PATH = REPOSITORY_ROOT / "asset/robot/bumi3/mjcf/bumi3_retarget.xml"


def test_bumi3_full_trajectory_qp_enforces_joint_contract_without_root_changes() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frame_count = 45
    fps = 30.0
    source = np.broadcast_to(model.qpos0, (frame_count, model.nq)).copy()
    names, addresses, lower, upper = scalar_joint_contract(model)
    source[:, addresses] = 0.5 * (lower + upper)

    # 造出逐帧 IK 常见的换分支/撞限位尖峰；根平移和根四元数另设连续信号，
    # 以证明关节 QP 不会顺带改变地板或根运动。
    source[:, 0] = np.linspace(-0.2, 0.3, frame_count)
    source[:, 2] = np.linspace(0.65, 0.72, frame_count)
    source[:, 3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    name_to_column = {name: index for index, name in enumerate(names)}
    source[20, addresses[name_to_column["l_arm_yaw_joint"]]] = upper[
        name_to_column["l_arm_yaw_joint"]
    ]
    source[24, addresses[name_to_column["r_arm_roll_joint"]]] = lower[
        name_to_column["r_arm_roll_joint"]
    ]
    source[28, addresses[name_to_column["waist_yaw_joint"]]] = upper[
        name_to_column["waist_yaw_joint"]
    ]

    optimized, diagnostics = optimize_scalar_joint_trajectory(
        model, source, fps, config["trajectory_qp"]
    )

    np.testing.assert_allclose(optimized[:, :7], source[:, :7], atol=0.0, rtol=0.0)
    assert diagnostics["enabled"] is True
    assert diagnostics["constraints_passed"] is True
    assert len(diagnostics["solver_records"]) == 21
    values = optimized[:, addresses]
    safe_lower = np.asarray(diagnostics["safe_lower_rad"])
    safe_upper = np.asarray(diagnostics["safe_upper_rad"])
    velocity_limits = np.asarray(diagnostics["velocity_limits_rad_s"])
    acceleration_limits = np.asarray(diagnostics["acceleration_limits_rad_s2"])
    jerk_limits = np.asarray(diagnostics["jerk_limits_rad_s3"])
    tolerance = float(diagnostics["constraint_violation"]["tolerance"])
    assert np.all(values >= safe_lower - tolerance)
    assert np.all(values <= safe_upper + tolerance)
    assert np.all(np.abs(np.diff(values, axis=0) * fps) <= velocity_limits + tolerance)
    assert np.all(
        np.abs(np.diff(values, n=2, axis=0) * fps**2)
        <= acceleration_limits + tolerance
    )
    assert np.all(
        np.abs(np.diff(values, n=3, axis=0) * fps**3) <= jerk_limits + tolerance
    )
    assert velocity_limits[name_to_column["waist_yaw_joint"]] == pytest.approx(9.0)
    for name in names:
        if name != "waist_yaw_joint":
            assert velocity_limits[name_to_column[name]] == pytest.approx(12.0)


def test_temporal_posture_vector_only_regularizes_bumi3_arms() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    vector, named = resolve_posture_cost_vector(
        model,
        config["temporal_posture_cost"],
        config["temporal_posture_joint_costs"],
    )
    expected = set(config["temporal_posture_joint_costs"])
    assert {name for name, cost in named.items() if cost > 0.0} == expected
    assert np.count_nonzero(vector) == 8
    for name in expected:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert vector[int(model.jnt_dofadr[joint_id])] == pytest.approx(5.0)
    assert np.all(vector[:6] == 0.0)

    with pytest.raises(ValueError, match="未知关节"):
        resolve_posture_cost_vector(model, 0.0, {"not_a_joint": 1.0})
