"""整段关节轨迹稀疏 QP 优化器。

本模块用于把逐帧 IK 产生的标量关节序列转换为一个同时满足位置限位、速度、
加速度和 jerk 约束的连续轨迹。它与传统的“先逐帧限速、最后再裁剪到关节限位”
不同：四类边界在同一个全时域凸二次规划中联合求解，因此不会因为最后一次位置
裁剪而突然把关节速度清零。目标函数保留原始 IK 姿态，并对一至三阶时间差分施加
软正则；所有物理边界均为硬约束。MuJoCo freejoint（根平移和根四元数）不会被本
模块改写，避免把地板对齐、根运动和执行器关节合同混在一起。

求解器按标量 hinge/slide 关节分别建立长度为 T 的稀疏 QP。各关节虽然独立求解，
但每个 QP 都覆盖完整序列而不是逐帧递推；这样既能保持问题规模可控，也能让尖峰
在前后帧之间连续分摊。输出包含逐关节求解状态、真实导数峰值和约束余量，供元数据
与交付验证器再次独立审计。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import mujoco
import numpy as np
import osqp
from scipy import sparse


def difference_matrix(frame_count: int, order: int) -> sparse.csc_matrix:
    """返回长度 ``frame_count`` 序列的前向 ``order`` 阶差分矩阵。"""
    if frame_count < 1:
        raise ValueError(f"frame_count 必须为正整数，实际 {frame_count}")
    if order not in {1, 2, 3}:
        raise ValueError(f"order 只支持 1/2/3，实际 {order}")
    rows = frame_count - order
    if rows <= 0:
        return sparse.csc_matrix((0, frame_count), dtype=np.float64)
    coefficients = {
        1: (-1.0, 1.0),
        2: (1.0, -2.0, 1.0),
        3: (-1.0, 3.0, -3.0, 1.0),
    }[order]
    diagonals = [
        np.full(rows, coefficient, dtype=np.float64)
        for coefficient in coefficients
    ]
    return sparse.diags(
        diagonals,
        offsets=tuple(range(order + 1)),
        shape=(rows, frame_count),
        format="csc",
    )


def scalar_joint_contract(
    model: mujoco.MjModel,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """按 MuJoCo 顺序返回标量关节名称、qpos 地址和位置上下界。"""
    names: list[str] = []
    addresses: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] not in {
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        }:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            raise ValueError(f"标量关节缺少名称: joint_id={joint_id}")
        if not bool(model.jnt_limited[joint_id]):
            raise ValueError(f"整段轨迹 QP 要求标量关节具有有限位置限位: {name}")
        joint_lower, joint_upper = (
            float(value) for value in model.jnt_range[joint_id]
        )
        if not (
            np.isfinite(joint_lower)
            and np.isfinite(joint_upper)
            and joint_lower < joint_upper
        ):
            raise ValueError(
                f"关节位置限位非法: {name}=[{joint_lower}, {joint_upper}]"
            )
        names.append(name)
        addresses.append(int(model.jnt_qposadr[joint_id]))
        lower.append(joint_lower)
        upper.append(joint_upper)
    return (
        names,
        np.asarray(addresses, dtype=np.int32),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )


def resolve_named_limits(
    names: Sequence[str],
    default_value: float,
    overrides: Mapping[str, Any] | None,
    label: str,
    *,
    allow_zero: bool = False,
) -> np.ndarray:
    """把默认值与按关节覆盖项解析成严格按 ``names`` 排列的数组。"""
    name_to_index = {str(name): index for index, name in enumerate(names)}
    if len(name_to_index) != len(names):
        raise ValueError(f"{label} 的关节名称存在重复")
    default_value = float(default_value)
    valid_default = default_value >= 0.0 if allow_zero else default_value > 0.0
    if not np.isfinite(default_value) or not valid_default:
        comparison = "非负" if allow_zero else "正"
        raise ValueError(f"{label}.default 必须为{comparison}有限值: {default_value}")
    values = np.full(len(names), default_value, dtype=np.float64)
    unknown = sorted(set(str(key) for key in (overrides or {})) - set(name_to_index))
    if unknown:
        raise ValueError(f"{label} 包含未知关节: {unknown}")
    for raw_name, raw_value in (overrides or {}).items():
        value = float(raw_value)
        valid_value = value >= 0.0 if allow_zero else value > 0.0
        if not np.isfinite(value) or not valid_value:
            comparison = "非负" if allow_zero else "正"
            raise ValueError(
                f"{label}.{raw_name} 必须为{comparison}有限值: {value}"
            )
        values[name_to_index[str(raw_name)]] = value
    return values


def resolve_posture_cost_vector(
    model: mujoco.MjModel,
    default_cost: float,
    joint_costs: Mapping[str, Any] | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """生成 Mink PostureTask 使用的 nv 维权重，仅覆盖命名标量关节。"""
    names, _, _, _ = scalar_joint_contract(model)
    costs = resolve_named_limits(
        names,
        default_cost,
        joint_costs,
        "temporal_posture_joint_costs",
        allow_zero=True,
    )
    cost_vector = np.zeros(model.nv, dtype=np.float64)
    named_costs: dict[str, float] = {}
    for name, cost in zip(names, costs, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        cost_vector[int(model.jnt_dofadr[joint_id])] = float(cost)
        named_costs[name] = float(cost)
    return cost_vector, named_costs


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _trajectory_metrics(values: np.ndarray, fps: float) -> dict[str, float]:
    velocity = np.diff(values, axis=0) * fps
    acceleration = np.diff(values, n=2, axis=0) * fps**2
    jerk = np.diff(values, n=3, axis=0) * fps**3
    return {
        "max_abs_velocity_rad_s": _max_abs(velocity),
        "max_abs_acceleration_rad_s2": _max_abs(acceleration),
        "max_abs_jerk_rad_s3": _max_abs(jerk),
    }


def optimize_scalar_joint_trajectory(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: float,
    config: Mapping[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """执行整段标量关节 QP，并返回优化 qpos 与可序列化审计数据。"""
    settings = dict(config or {})
    enabled = bool(settings.get("enabled", False))
    source = np.asarray(qpos, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != model.nq:
        raise ValueError(
            f"qpos 必须为 [T,{model.nq}]，实际 {source.shape}"
        )
    if not np.all(np.isfinite(source)):
        raise ValueError("整段轨迹 QP 输入包含 NaN/Inf")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps 必须为正有限值，实际 {fps}")
    if not enabled:
        return source.copy(), {"enabled": False}

    solver_name = str(settings.get("solver", "osqp")).lower()
    if solver_name != "osqp":
        raise ValueError(f"trajectory_qp.solver 只支持 osqp，实际 {solver_name}")
    frame_count = int(source.shape[0])
    names, addresses, lower, upper = scalar_joint_contract(model)
    if not names:
        raise ValueError("整段轨迹 QP 没有找到标量关节")

    position_margin = float(settings.get("position_limit_margin_rad", 1.0e-3))
    if not np.isfinite(position_margin) or position_margin < 0.0:
        raise ValueError(
            "trajectory_qp.position_limit_margin_rad 必须为非负有限值"
        )
    margin = np.minimum(position_margin, 0.05 * (upper - lower))
    safe_lower = lower + margin
    safe_upper = upper - margin
    if np.any(safe_lower >= safe_upper):
        bad = [names[index] for index in np.flatnonzero(safe_lower >= safe_upper)]
        raise ValueError(f"位置限位安全裕度过大: {bad}")

    velocity_limits = resolve_named_limits(
        names,
        settings.get("velocity_limit_default_rad_s", 12.0),
        settings.get("velocity_limit_overrides_rad_s"),
        "trajectory_qp.velocity_limit",
    )
    acceleration_limits = resolve_named_limits(
        names,
        settings.get("acceleration_limit_default_rad_s2", 80.0),
        settings.get("acceleration_limit_overrides_rad_s2"),
        "trajectory_qp.acceleration_limit",
    )
    jerk_limits = resolve_named_limits(
        names,
        settings.get("jerk_limit_default_rad_s3", 2400.0),
        settings.get("jerk_limit_overrides_rad_s3"),
        "trajectory_qp.jerk_limit",
    )

    track_weight = float(settings.get("track_weight", 1.0))
    velocity_weight = float(settings.get("velocity_smoothing_weight", 0.1))
    acceleration_weight = float(
        settings.get("acceleration_smoothing_weight", 1.0)
    )
    jerk_weight = float(settings.get("jerk_smoothing_weight", 0.1))
    for label, value in (
        ("track_weight", track_weight),
        ("velocity_smoothing_weight", velocity_weight),
        ("acceleration_smoothing_weight", acceleration_weight),
        ("jerk_smoothing_weight", jerk_weight),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"trajectory_qp.{label} 必须为非负有限值: {value}")
    if track_weight <= 0.0:
        raise ValueError("trajectory_qp.track_weight 必须大于 0")

    d1 = difference_matrix(frame_count, 1)
    d2 = difference_matrix(frame_count, 2)
    d3 = difference_matrix(frame_count, 3)
    identity = sparse.eye(frame_count, format="csc", dtype=np.float64)
    hessian = track_weight * identity
    for weight, operator in (
        (velocity_weight, d1),
        (acceleration_weight, d2),
        (jerk_weight, d3),
    ):
        if weight > 0.0 and operator.shape[0] > 0:
            hessian = hessian + weight * (operator.T @ operator)
    p_matrix = sparse.triu(2.0 * hessian, format="csc")
    constraint_matrix = sparse.vstack(
        [identity, d1, d2, d3], format="csc"
    )

    eps_abs = float(settings.get("eps_abs", 1.0e-7))
    eps_rel = float(settings.get("eps_rel", 1.0e-7))
    max_iter = int(settings.get("max_iter", 50000))
    feasibility_tolerance = float(
        settings.get("feasibility_tolerance", 1.0e-4)
    )
    constraint_guard_fraction = float(
        settings.get("constraint_guard_fraction", 1.0e-3)
    )
    if min(eps_abs, eps_rel, feasibility_tolerance) <= 0.0 or max_iter <= 0:
        raise ValueError("trajectory_qp 求解精度、可行性容差和 max_iter 必须为正")
    if not np.isfinite(constraint_guard_fraction) or not (
        0.0 <= constraint_guard_fraction < 0.1
    ):
        raise ValueError(
            "trajectory_qp.constraint_guard_fraction 必须位于 [0,0.1)"
        )
    # OSQP 的原始/对偶残差是在经过自动缩放的离散问题中判定的；长序列三阶
    # 差分换回 rad/s^3 后会放大极小数值残差。求解时把导数边界向内收 0.1%，
    # 最终仍按用户声明的原始边界独立验收，而不是放宽验收容差。
    solver_velocity_limits = velocity_limits * (1.0 - constraint_guard_fraction)
    solver_acceleration_limits = acceleration_limits * (
        1.0 - constraint_guard_fraction
    )
    solver_jerk_limits = jerk_limits * (1.0 - constraint_guard_fraction)

    output = source.copy()
    raw_joint_values = source[:, addresses]
    target_joint_values = np.clip(raw_joint_values, safe_lower, safe_upper)
    solver_records: list[dict[str, Any]] = []
    for joint_index, joint_name in enumerate(names):
        lower_parts = [
            np.full(frame_count, safe_lower[joint_index], dtype=np.float64),
            np.full(d1.shape[0], -solver_velocity_limits[joint_index] / fps),
            np.full(
                d2.shape[0], -solver_acceleration_limits[joint_index] / fps**2
            ),
            np.full(d3.shape[0], -solver_jerk_limits[joint_index] / fps**3),
        ]
        upper_parts = [
            np.full(frame_count, safe_upper[joint_index], dtype=np.float64),
            np.full(d1.shape[0], solver_velocity_limits[joint_index] / fps),
            np.full(d2.shape[0], solver_acceleration_limits[joint_index] / fps**2),
            np.full(d3.shape[0], solver_jerk_limits[joint_index] / fps**3),
        ]
        linear = -2.0 * track_weight * target_joint_values[:, joint_index]
        solver = osqp.OSQP()
        solver.setup(
            P=p_matrix,
            q=linear,
            A=constraint_matrix,
            l=np.concatenate(lower_parts),
            u=np.concatenate(upper_parts),
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            polishing=False,
            verbose=False,
        )
        result = solver.solve(raise_error=True)
        status = str(result.info.status)
        if result.x is None or int(result.info.status_val) not in {1, 2}:
            raise RuntimeError(
                f"整段轨迹 QP 求解失败: joint={joint_name}, status={status}"
            )
        solution = np.asarray(result.x, dtype=np.float64)
        if solution.shape != (frame_count,) or not np.all(np.isfinite(solution)):
            raise RuntimeError(
                f"整段轨迹 QP 返回非法解: joint={joint_name}, shape={solution.shape}"
            )
        output[:, addresses[joint_index]] = solution
        solver_records.append(
            {
                "joint_name": joint_name,
                "status": status,
                "iterations": int(result.info.iter),
                "objective": float(result.info.obj_val),
            }
        )

    optimized_joint_values = output[:, addresses]
    velocity = np.diff(optimized_joint_values, axis=0) * fps
    acceleration = np.diff(optimized_joint_values, n=2, axis=0) * fps**2
    jerk = np.diff(optimized_joint_values, n=3, axis=0) * fps**3
    position_violation = max(
        _max_abs(np.minimum(optimized_joint_values - safe_lower, 0.0)),
        _max_abs(np.maximum(optimized_joint_values - safe_upper, 0.0)),
    )
    velocity_violation = _max_abs(
        np.maximum(np.abs(velocity) - velocity_limits, 0.0)
    )
    acceleration_violation = _max_abs(
        np.maximum(np.abs(acceleration) - acceleration_limits, 0.0)
    )
    jerk_violation = _max_abs(np.maximum(np.abs(jerk) - jerk_limits, 0.0))
    maximum_violation = max(
        position_violation,
        velocity_violation,
        acceleration_violation,
        jerk_violation,
    )
    if maximum_violation > feasibility_tolerance:
        raise RuntimeError(
            "整段轨迹 QP 解违反硬约束: "
            f"position={position_violation}, velocity={velocity_violation}, "
            f"acceleration={acceleration_violation}, jerk={jerk_violation}, "
            f"tolerance={feasibility_tolerance}"
        )

    change = optimized_joint_values - raw_joint_values
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "solver": solver_name,
        "frame_count": frame_count,
        "joint_names": names,
        "joint_qpos_addresses": addresses.tolist(),
        "position_limit_margin_rad": position_margin,
        "constraint_guard_fraction": constraint_guard_fraction,
        "safe_lower_rad": safe_lower.tolist(),
        "safe_upper_rad": safe_upper.tolist(),
        "velocity_limits_rad_s": velocity_limits.tolist(),
        "acceleration_limits_rad_s2": acceleration_limits.tolist(),
        "jerk_limits_rad_s3": jerk_limits.tolist(),
        "weights": {
            "track": track_weight,
            "velocity_difference": velocity_weight,
            "acceleration_difference": acceleration_weight,
            "jerk_difference": jerk_weight,
        },
        "solver_records": solver_records,
        "raw_metrics": _trajectory_metrics(raw_joint_values, fps),
        "optimized_metrics": _trajectory_metrics(optimized_joint_values, fps),
        "max_abs_joint_change_rad": _max_abs(change),
        "mean_abs_joint_change_rad": float(np.mean(np.abs(change))),
        "constraint_violation": {
            "position_rad": position_violation,
            "velocity_rad_s": velocity_violation,
            "acceleration_rad_s2": acceleration_violation,
            "jerk_rad_s3": jerk_violation,
            "maximum": maximum_violation,
            "tolerance": feasibility_tolerance,
        },
        "constraints_passed": bool(maximum_violation <= feasibility_tolerance),
    }
    return output, diagnostics
