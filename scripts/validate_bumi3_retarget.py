#!/usr/bin/env python3
"""验证 BUMI3 重定向资产、qpos CSV、接触质量与 IsaacLab Mimic NPZ。

脚本既可只做模型预检，也可对完整的 keypoint/CSV/metadata/NPZ 交付边界做联合
检查。模型部分核对唯一 freejoint、21 个驱动关节、初始姿态、marker、ground、
连杆长度与左右对称性；动作部分通过逐帧 MuJoCo FK 统计任务位置/旋转误差、
关节限位裕度、速度/加速度/jerk、根四元数和脚跟/脚尖接触区间的滑移及穿地；
NPZ 部分严格核对配置声明的输出 fps、21 关节、22 body、有限值和 wxyz 单位四元数。所有
结果写入结构化 JSON，任何硬契约或硬质量阈值失败都会返回非零退出码，便于一键
流水线在错误发生处立即停止。关节贴限位占比可按配置选择 hard 或 warning；无论
采用哪一种，实际越限以及速度、加速度、jerk 门槛始终是硬验收项。
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bumi3_common import (
    BUMI3_ISAAC_BODY_NAMES,
    BUMI3_JOINT_NAMES,
    BUMI3_MARKER_BODY_NAMES,
    joint_qpos_addresses,
    load_yaml,
    resolve_config_path,
    sha256_file,
)
from export_bumi3_mimic_npz import load_csv_qpos


# 这些门槛用于离线重定向的可视化质量回归，不代表实机速度或力矩安全边界。
# 位置误差同时限制所有有效任务的整体 RMS 与最差单任务 RMS；关节部分同时限制
# 全部样本的限位邻近率、任一关节精确贴边率以及真实相邻帧速度，避免旧配置中
# “整体平均尚可，但单只手偏离或单个关节长期卡死”的问题再次被漏检。
MAX_AGGREGATE_TASK_POSITION_RMS_M = 0.075
MAX_SINGLE_TASK_POSITION_RMS_M = 0.12
MAX_CALF_TASK_POSITION_RMS_M = 0.18
MAX_JOINT_LIMIT_NEAR_RATE = 0.15
MAX_SINGLE_JOINT_EXACT_LIMIT_RATE = 0.65
MAX_FRAME_TO_FRAME_JOINT_VELOCITY_RAD_S = 35.0
MAX_P99_JOINT_ACCELERATION_RAD_S2 = 650.0
MAX_ABS_JOINT_ACCELERATION_RAD_S2 = 2000.0
MAX_P99_JOINT_JERK_RAD_S3 = 50000.0
MAX_ABS_JOINT_JERK_RAD_S3 = 150000.0
MAX_SUPPORT_POINT_HEIGHT_ABOVE_GROUND_M = 0.02
MAX_FLAT_HEEL_TOE_HEIGHT_DIFFERENCE_M = 0.01
MAX_STATIC_FLOOR_P01_ERROR_M = 0.02
MIN_STATIC_ROOT_HEIGHT_M = 0.15


def effective_joint_range(
    model: mujoco.MjModel, config: dict[str, Any], joint_name: str
) -> tuple[float, float]:
    """返回包含配置收紧量的实际 IK 关节限位。"""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    lower, upper = (float(value) for value in model.jnt_range[joint_id])
    for name_pattern, raw_offsets in config.get(
        "joints_limit_offset_degrees", {}
    ).items():
        if str(name_pattern) not in joint_name:
            continue
        if isinstance(raw_offsets, (list, tuple)):
            if len(raw_offsets) != 2:
                raise ValueError(
                    "joints_limit_offset_degrees."
                    f"{name_pattern} 必须是 [lower, upper]"
                )
            lower_offset, upper_offset = raw_offsets
        else:
            lower_offset, upper_offset = raw_offsets, 0.0
        lower += float(np.deg2rad(float(lower_offset)))
        upper += float(np.deg2rad(float(upper_offset)))
    return (lower, upper) if lower <= upper else (upper, lower)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 BUMI3 模型和重定向产物")
    parser.add_argument(
        "--config", type=Path, default=Path("config/robot/bumi3.yaml"), help="BUMI3 配置"
    )
    parser.add_argument("--keypoints", type=Path, default=None, help="keypoint PKL")
    parser.add_argument("--csv", type=Path, default=None, help="qpos CSV")
    parser.add_argument("--metadata", type=Path, default=None, help="CSV 元数据 JSON")
    parser.add_argument("--npz", type=Path, default=None, help="Mimic NPZ")
    parser.add_argument("--report", type=Path, required=True, help="输出 JSON 报告")
    return parser.parse_args()


class ValidationReport:
    """收集通过项、警告、失败和数值指标，并保证失败时仍能落盘。"""

    def __init__(self) -> None:
        self.checks: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.failures: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)

    def payload(self) -> dict[str, Any]:
        return {
            "status": "passed" if not self.failures else "failed",
            "checks": self.checks,
            "warnings": self.warnings,
            "failures": self.failures,
        }


def named_body_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body 不存在: expected={name}")
    return np.asarray(data.xpos[body_id], dtype=np.float64)


def validate_model(
    config: dict[str, Any], config_path: Path, report: ValidationReport
) -> tuple[mujoco.MjModel, Path]:
    """验证静态模型合同并返回已加载模型。"""
    xml_path = resolve_config_path(config_path, str(config["robot_xml_path"]))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    free_ids = [
        idx
        for idx in range(model.njnt)
        if model.jnt_type[idx] == mujoco.mjtJoint.mjJNT_FREE
    ]
    report.require(len(free_ids) == 1, f"freejoint 数量必须为 1，实际 {len(free_ids)}")
    joint_names = [str(value) for value in config.get("isaac_joint_names", [])]
    body_names = [str(value) for value in config.get("isaac_body_names", [])]
    report.require(joint_names == BUMI3_JOINT_NAMES, "Isaac 21 关节顺序与 BUMI3 契约不一致")
    report.require(body_names == BUMI3_ISAAC_BODY_NAMES, "Isaac 22 body 顺序与 BUMI3 契约不一致")
    addresses = joint_qpos_addresses(model, BUMI3_JOINT_NAMES)
    invalid_ranges = []
    for name in BUMI3_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower, upper = model.jnt_range[joint_id]
        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            invalid_ranges.append(name)
    report.require(not invalid_ranges, f"关节限位非法: {invalid_ranges}")
    right_roll_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "r_arm_roll_joint"
    )
    right_roll_range = np.asarray(model.jnt_range[right_roll_id], dtype=np.float64)
    report.require(
        np.allclose(right_roll_range, [-1.94, 0.14], atol=1.0e-9),
        f"右臂 roll 限位错误: expected=[-1.94,0.14], actual={right_roll_range.tolist()}",
    )

    qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
    root = config.get("initial_root_pose")
    if root and len(free_ids) == 1:
        root_addr = int(model.jnt_qposadr[free_ids[0]])
        position = np.asarray(root.get("position"), dtype=np.float64)
        quat = np.asarray(root.get("quaternion_wxyz"), dtype=np.float64)
        report.require(position.shape == (3,), f"初始根位置 shape 错误: {position.shape}")
        report.require(quat.shape == (4,), f"初始根四元数 shape 错误: {quat.shape}")
        if position.shape == (3,) and quat.shape == (4,) and np.linalg.norm(quat) > 0:
            qpos[root_addr : root_addr + 3] = position
            qpos[root_addr + 3 : root_addr + 7] = quat / np.linalg.norm(quat)
    initial_violations = []
    for name, raw_value in config.get("initial_joint_positions", {}).items():
        if name not in addresses:
            initial_violations.append(f"missing:{name}")
            continue
        value = float(raw_value)
        lower, upper = effective_joint_range(model, config, name)
        if value < lower - 1.0e-9 or value > upper + 1.0e-9:
            initial_violations.append(f"{name}:{value} not in [{lower},{upper}]")
        qpos[addresses[name]] = value
    report.require(not initial_violations, f"初始关节姿态非法: {initial_violations}")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    missing_markers = [
        name
        for name in BUMI3_MARKER_BODY_NAMES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0
    ]
    report.require(not missing_markers, f"缺少 marker body: {missing_markers}")
    marker_contract = {}
    for name in BUMI3_MARKER_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        geom_start = int(model.body_geomadr[body_id])
        geom_count = int(model.body_geomnum[body_id])
        geom_ids = range(geom_start, geom_start + geom_count)
        collision_free = all(
            model.geom_contype[idx] == 0 and model.geom_conaffinity[idx] == 0
            for idx in geom_ids
        )
        opaque_red = bool(geom_count > 0) and all(
            np.allclose(model.geom_rgba[idx], [1.0, 0.0, 0.0, 1.0])
            for idx in geom_ids
        )
        marker_contract[name] = {
            "mass": float(model.body_mass[body_id]),
            "collision_free": bool(collision_free),
            "opaque_red": opaque_red,
        }
        report.require(model.body_mass[body_id] == 0.0, f"marker 不应产生质量: {name}")
        report.require(collision_free, f"marker 不应参与碰撞: {name}")
        if name in {
            "left_foot_end_link",
            "left_toe_link",
            "right_foot_end_link",
            "right_toe_link",
        }:
            report.require(opaque_red, f"足底 marker 必须是不透明红色: {name}")

    spec = mujoco.MjSpec.from_file(str(xml_path))
    ground_count = sum(1 for geom in spec.geoms if geom.name == "ground")
    report.require(ground_count == 1, f"ground 数量必须为 1，实际 {ground_count}")

    # 连杆尺寸与 marker 前后/高度关系属于静态资产合同，不应随用于 IK 热启动的
    # initial_joint_positions 改变。恢复 MJCF qpos0 后再测几何；初始姿态本身的
    # 名称、数值与限位合法性已经在上方独立验证。
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    link_lengths = {}
    for semantic_name, endpoints in config.get("robot_links", {}).items():
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            report.failures.append(f"robot_links.{semantic_name} 必须有两个端点")
            continue
        length = float(
            np.linalg.norm(
                named_body_position(model, data, str(endpoints[1]))
                - named_body_position(model, data, str(endpoints[0]))
            )
        )
        link_lengths[semantic_name] = length
        report.require(length > 1.0e-6, f"robot link 长度非正: {semantic_name}={length}")
    symmetry = {}
    for left, right in (
        ("left_hip", "right_hip"),
        ("left_thigh", "right_thigh"),
        ("left_calf", "right_calf"),
        ("left_arm", "right_arm"),
        ("left_fore_arm", "right_fore_arm"),
    ):
        if left in link_lengths and right in link_lengths:
            relative = abs(link_lengths[left] - link_lengths[right]) / max(
                link_lengths[left], link_lengths[right]
            )
            symmetry[f"{left}:{right}"] = relative
            report.require(relative <= 0.05, f"左右连杆不对称超过 5%: {left}/{right}={relative}")

    marker_local_positions = {
        name: np.asarray(
            model.body_pos[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            ],
            dtype=np.float64,
        )
        for name in BUMI3_MARKER_BODY_NAMES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    }
    for side in ("left", "right"):
        heel = marker_local_positions.get(f"{side}_foot_end_link")
        toe = marker_local_positions.get(f"{side}_toe_link")
        if heel is not None and toe is not None:
            report.require(toe[0] > heel[0], f"{side} toe 必须位于 heel 前方")
            report.require(abs(toe[2] - heel[2]) < 0.01, f"{side} heel/toe 高度差过大")

    # 方向标定参考姿态必须独立于 IK 热启动姿态；这里显式加载 solved pose，确保
    # 标定所定义的中立脚严格平行地面，即便 initial_joint_positions 为稳定屈膝而
    # 保留非零踝角，也不能再污染方向标定合同。
    reference_joint_positions = config.get("reference_pose", {}).get(
        "solved_joint_positions", {}
    )
    reference_foot_height_differences = {}
    if reference_joint_positions:
        reference_qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
        for name, raw_value in reference_joint_positions.items():
            if name in addresses:
                reference_qpos[addresses[name]] = float(raw_value)
        data.qpos[:] = reference_qpos
        mujoco.mj_forward(model, data)
        for side in ("left", "right"):
            difference = abs(
                float(named_body_position(model, data, f"{side}_toe_link")[2])
                - float(named_body_position(model, data, f"{side}_foot_end_link")[2])
            )
            reference_foot_height_differences[side] = difference
            report.require(
                difference <= 1.0e-5,
                f"方向标定参考姿态的 {side} 脚不平: heel/toe dz={difference}",
            )
    else:
        report.failures.append("reference_pose.solved_joint_positions 缺失")

    report.checks["model"] = {
        "xml": str(xml_path),
        "nq": int(model.nq),
        "joint_count": len(addresses),
        "body_count": len(body_names),
        "ground_count": ground_count,
        "right_arm_roll_range": right_roll_range.tolist(),
        "marker_contract": marker_contract,
        "robot_link_lengths": link_lengths,
        "symmetry_relative_difference": symmetry,
        "reference_foot_height_difference_m": reference_foot_height_differences,
    }
    return model, xml_path


def rotation_error_radians(actual_wxyz: np.ndarray, target_wxyz: np.ndarray) -> float:
    """返回两个单位四元数之间忽略符号的最短旋转角。"""
    actual = actual_wxyz / np.linalg.norm(actual_wxyz)
    target = target_wxyz / np.linalg.norm(target_wxyz)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(actual, target)), 0.0, 1.0)))


def contact_intervals(states: np.ndarray) -> list[tuple[int, int]]:
    """将布尔序列转换为半开区间 [start,end)。"""
    intervals = []
    start = None
    for idx, active in enumerate(np.r_[states.astype(bool), False]):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            intervals.append((start, idx))
            start = None
    return intervals


def validate_motion(
    model: mujoco.MjModel,
    config: dict[str, Any],
    config_path: Path,
    keypoints_path: Path,
    csv_path: Path,
    metadata_path: Path | None,
    report: ValidationReport,
) -> int:
    """联合验证 CSV、keypoint 接触与 FK 任务误差，返回帧数。"""
    qpos = load_csv_qpos(csv_path, model)
    with keypoints_path.expanduser().resolve().open("rb") as file:
        keypoints = pickle.load(file)
    positions = np.asarray(keypoints["positions"], dtype=np.float64)
    quaternions = np.asarray(keypoints["quaternions"], dtype=np.float64)
    contact_states = np.asarray(keypoints["contact_states"], dtype=bool)
    contact_names = [str(value) for value in keypoints["contact_names"]]
    keypoint_names = [str(value) for value in keypoints["keypoint_names"]]
    fps = float(keypoints["fps"])
    expected_fps = float(config["output"]["target_fps"])
    expected_ground_references = [
        str(value)
        for value in config.get("ground_reference_contacts", contact_names)
    ]
    actual_ground_references = [
        str(value) for value in keypoints.get("ground_reference_contacts", [])
    ]
    expected_dynamic_height_offset = bool(
        config.get("contact_height_dynamic_offset_enabled", True)
    )
    expected_relative_sequence_floor = bool(
        config.get("contact_height_relative_to_sequence_floor", False)
    )
    expected_height_offset_mode = (
        "dynamic_contact"
        if expected_dynamic_height_offset
        else (
            "static_sequence_floor"
            if expected_relative_sequence_floor
            else "disabled"
        )
    )
    actual_dynamic_height_offset = keypoints.get(
        "contact_height_dynamic_offset_enabled"
    )
    actual_height_offset_mode = keypoints.get("contact_height_offset_mode")
    expected_floor_method = (
        str(config.get("contact_height_floor_method", "percentile"))
        if expected_relative_sequence_floor
        else "disabled"
    )
    actual_floor_method = keypoints.get("contact_height_floor_method")
    source_floor_fit = keypoints.get("contact_height_source_floor_fit")
    retarget_floor_fit = keypoints.get("contact_height_retarget_floor_fit")
    height_offset_min = float(keypoints.get("contact_height_offset_min", np.nan))
    height_offset_median = float(
        keypoints.get("contact_height_offset_median", np.nan)
    )
    height_offset_max = float(keypoints.get("contact_height_offset_max", np.nan))
    frame_count = qpos.shape[0]
    report.require(positions.shape == (frame_count, len(keypoint_names), 3), "keypoint 位置 shape 与 CSV 不一致")
    report.require(quaternions.shape == (frame_count, len(keypoint_names), 4), "keypoint 四元数 shape 与 CSV 不一致")
    report.require(contact_states.shape == (frame_count, len(contact_names)), "接触状态 shape 与 CSV 不一致")
    report.require(
        np.isfinite(expected_fps) and expected_fps > 0.0,
        f"配置 output.target_fps 必须为正有限值，实际 {expected_fps}",
    )
    report.require(
        abs(fps - expected_fps) < 1.0e-9,
        f"动作 fps 与配置不一致: expected={expected_fps}, actual={fps}",
    )
    report.require(
        actual_ground_references == expected_ground_references,
        "keypoint ground reference 与配置不一致: "
        f"expected={expected_ground_references}, actual={actual_ground_references}",
    )
    report.require(
        isinstance(actual_dynamic_height_offset, (bool, np.bool_))
        and bool(actual_dynamic_height_offset) == expected_dynamic_height_offset,
        "keypoint 动态全身 Z 偏移策略与配置不一致: "
        f"expected={expected_dynamic_height_offset}, "
        f"actual={actual_dynamic_height_offset}",
    )
    report.require(
        actual_height_offset_mode == expected_height_offset_mode,
        "keypoint 高度偏移模式与配置不一致: "
        f"expected={expected_height_offset_mode}, actual={actual_height_offset_mode}",
    )
    report.require(
        actual_floor_method == expected_floor_method,
        "keypoint 地板估计方法与配置不一致: "
        f"expected={expected_floor_method}, actual={actual_floor_method}",
    )
    if expected_floor_method == "stable_support_dense_median":
        minimum_floor_samples = int(config["contact_height_floor_fit_min_samples"])
        for label, floor_fit in (
            ("source", source_floor_fit),
            ("retarget", retarget_floor_fit),
        ):
            report.require(
                isinstance(floor_fit, dict)
                and floor_fit.get("method") == expected_floor_method,
                f"{label} 稳健地板拟合报告缺失或方法错误: {floor_fit}",
            )
            if isinstance(floor_fit, dict):
                report.require(
                    int(floor_fit.get("stable_sample_count", -1))
                    >= minimum_floor_samples,
                    f"{label} 稳定足点样本不足: {floor_fit}",
                )
                report.require(
                    int(floor_fit.get("inlier_sample_count", -1))
                    >= minimum_floor_samples,
                    f"{label} 稳健地板内点不足: {floor_fit}",
                )
                report.require(
                    np.isfinite(float(floor_fit.get("floor_height_m", np.nan))),
                    f"{label} 稳健地板高度不是有限值: {floor_fit}",
                )
    if not expected_dynamic_height_offset:
        height_offset_values = np.asarray(
            [height_offset_min, height_offset_median, height_offset_max],
            dtype=np.float64,
        )
        report.require(
            bool(np.all(np.isfinite(height_offset_values)))
            and float(np.ptp(height_offset_values)) < 1.0e-9,
            "禁用动态全身 Z 偏移时，keypoint 高度偏移必须整段恒定: "
            f"actual={height_offset_values.tolist()}",
        )
        expected_constant_height = (
            float(keypoints.get("contact_height_retarget_sequence_floor", np.nan))
            if expected_relative_sequence_floor
            else 0.0
        )
        report.require(
            np.isfinite(expected_constant_height)
            and bool(
                np.allclose(
                    height_offset_values,
                    expected_constant_height,
                    rtol=0.0,
                    atol=1.0e-6,
                )
            ),
            "静态高度偏移必须等于四足点序列地板标定值: "
            f"expected={expected_constant_height}, actual={height_offset_values.tolist()}",
        )
    finite_count = int(np.size(qpos) - np.count_nonzero(np.isfinite(qpos)))
    root_norm_error = float(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.0)))
    report.require(finite_count == 0, f"CSV NaN/Inf 数量为 {finite_count}")
    report.require(root_norm_error < 1.0e-5, f"根四元数范数误差过大: {root_norm_error}")

    addresses = joint_qpos_addresses(model, BUMI3_JOINT_NAMES)
    joint_pos = np.stack([qpos[:, addresses[name]] for name in BUMI3_JOINT_NAMES], axis=1)
    violation_count = 0
    near_count = 0
    per_joint_limits = {}
    for joint_idx, name in enumerate(BUMI3_JOINT_NAMES):
        lower, upper = effective_joint_range(model, config, name)
        values = joint_pos[:, joint_idx]
        violations = (values < lower - 1.0e-6) | (values > upper + 1.0e-6)
        violation_count += int(np.count_nonzero(violations))
        margin = 0.05 * (upper - lower)
        near = (values - lower < margin) | (upper - values < margin)
        exact = (values - lower < 1.0e-5) | (upper - values < 1.0e-5)
        near_count += int(np.count_nonzero(near))
        per_joint_limits[name] = {
            "effective_range_rad": [lower, upper],
            "near_limit_rate": float(np.mean(near)),
            "exact_limit_rate": float(np.mean(exact)),
        }
    # BUMI3 的 21 个驱动关节全是有限位 hinge，不是可跨 ±pi 连续旋转的关节。
    # np.unwrap 会把真实的限位间跳变误判为角度环绕；中心差分也会把单帧尖峰分摊
    # 到相邻两帧。这里直接逐阶相邻帧做差，报告真实的最坏离散速度/加速度/jerk。
    joint_vel = np.diff(joint_pos, axis=0) * fps
    joint_acc = np.diff(joint_vel, axis=0) * fps
    joint_jerk = np.diff(joint_acc, axis=0) * fps

    def max_abs(values: np.ndarray) -> float:
        return float(np.max(np.abs(values))) if values.size else 0.0

    def p99_abs(values: np.ndarray) -> float:
        return float(np.percentile(np.abs(values), 99)) if values.size else 0.0

    joint_limit_near_rate = float(near_count / joint_pos.size)
    maximum_exact_limit_rate = max(
        values["exact_limit_rate"] for values in per_joint_limits.values()
    )
    maximum_joint_velocity = max_abs(joint_vel)
    p99_joint_acceleration = p99_abs(joint_acc)
    maximum_joint_acceleration = max_abs(joint_acc)
    p99_joint_jerk = p99_abs(joint_jerk)
    maximum_joint_jerk = max_abs(joint_jerk)
    joint_limit_occupancy_validation = str(
        config.get("joint_limit_occupancy_validation", "hard")
    )
    report.require(
        joint_limit_occupancy_validation in {"hard", "warning"},
        "joint_limit_occupancy_validation 必须是 hard 或 warning，实际 "
        f"{joint_limit_occupancy_validation}",
    )
    report.require(violation_count == 0, f"关节限位违反次数为 {violation_count}")
    occupancy_messages = []
    if joint_limit_near_rate >= MAX_JOINT_LIMIT_NEAR_RATE:
        occupancy_messages.append(f"关节限位邻近率超阈值: {joint_limit_near_rate}")
    if maximum_exact_limit_rate >= MAX_SINGLE_JOINT_EXACT_LIMIT_RATE:
        occupancy_messages.append(
            f"单关节精确贴限位率超阈值: {maximum_exact_limit_rate}"
        )
    if joint_limit_occupancy_validation == "hard":
        report.failures.extend(occupancy_messages)
    else:
        report.warnings.extend(
            f"连续性优先模式仅警告：{message}" for message in occupancy_messages
        )
    report.require(
        maximum_joint_velocity < MAX_FRAME_TO_FRAME_JOINT_VELOCITY_RAD_S,
        f"相邻帧关节速度超阈值: {maximum_joint_velocity}",
    )
    report.require(
        p99_joint_acceleration < MAX_P99_JOINT_ACCELERATION_RAD_S2,
        f"关节加速度 P99 超阈值: {p99_joint_acceleration}",
    )
    report.require(
        maximum_joint_acceleration < MAX_ABS_JOINT_ACCELERATION_RAD_S2,
        f"关节最大加速度超阈值: {maximum_joint_acceleration}",
    )
    report.require(
        p99_joint_jerk < MAX_P99_JOINT_JERK_RAD_S3,
        f"关节 jerk P99 超阈值: {p99_joint_jerk}",
    )
    report.require(
        maximum_joint_jerk < MAX_ABS_JOINT_JERK_RAD_S3,
        f"关节最大 jerk 超阈值: {maximum_joint_jerk}",
    )

    data = mujoco.MjData(model)
    keypoint_index = {name: idx for idx, name in enumerate(keypoint_names)}
    task_position_errors: dict[str, list[float]] = {
        name: [] for name in config.get("ik_match_table", {})
    }
    task_rotation_errors: dict[str, list[float]] = {
        name: [] for name in config.get("ik_match_table", {})
    }
    contact_frame_names = {
        str(source_name): str(entry["frame_name"])
        for source_name, entry in config.get("contact_map", {}).items()
        if bool(entry.get("enabled", True)) and bool(entry.get("lock_position", True))
    }
    robot_contact_frames = [str(value) for value in config.get("contact_links", [])]
    if len(robot_contact_frames) == len(contact_names):
        for source_name, frame_name in zip(contact_names, robot_contact_frames):
            if source_name in expected_ground_references:
                contact_frame_names.setdefault(source_name, frame_name)
    missing_ground_frames = [
        source_name
        for source_name in expected_ground_references
        if source_name not in contact_frame_names
    ]
    report.require(
        not missing_ground_frames,
        f"无法解析 ground reference 对应的机器人足点: missing={missing_ground_frames}",
    )
    contact_body_positions = {
        source_name: np.empty((frame_count, 3), dtype=np.float64)
        for source_name in contact_frame_names
    }
    for frame_idx, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        for source_name, entry in config.get("ik_match_table", {}).items():
            frame_name, position_cost, rotation_cost = entry
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
            source_idx = keypoint_index[source_name]
            if float(position_cost) != 0.0:
                task_position_errors[source_name].append(
                    float(np.linalg.norm(data.xpos[body_id] - positions[frame_idx, source_idx]))
                )
            if float(rotation_cost) != 0.0:
                task_rotation_errors[source_name].append(
                    rotation_error_radians(
                        np.asarray(data.xquat[body_id]), quaternions[frame_idx, source_idx]
                    )
                )
        for source_name in contact_body_positions:
            frame_name = contact_frame_names[source_name]
            contact_body_positions[source_name][frame_idx] = named_body_position(
                model, data, frame_name
            )

    task_rms = {}
    evaluated_position_errors: dict[str, np.ndarray] = {}
    for source_name in task_position_errors:
        all_position_errors = np.asarray(
            task_position_errors[source_name], dtype=np.float64
        )
        pos = all_position_errors
        evaluation_scope = "all_frames"
        if source_name in {"left_calf", "right_calf"} and pos.size == frame_count:
            side = source_name.split("_", 1)[0]
            heel_states = contact_states[:, contact_names.index(f"{side}_foot_end")]
            toe_states = contact_states[:, contact_names.index(f"{side}_toe")]
            pos = pos[~(heel_states | toe_states)]
            evaluation_scope = "swing_frames"
        evaluated_position_errors[source_name] = pos
        rot = np.asarray(task_rotation_errors[source_name], dtype=np.float64)
        task_rms[source_name] = {
            "position_rms_m": float(np.sqrt(np.mean(pos**2))) if pos.size else None,
            "position_rms_all_frames_m": (
                float(np.sqrt(np.mean(all_position_errors**2)))
                if all_position_errors.size
                else None
            ),
            "position_evaluation_scope": evaluation_scope,
            "position_evaluated_frame_count": int(pos.size),
            "rotation_rms_rad": float(np.sqrt(np.mean(rot**2))) if rot.size else None,
        }
    active_position_errors = [
        values for values in evaluated_position_errors.values() if values.size
    ]
    aggregate_position_rms = (
        float(np.sqrt(np.mean(np.concatenate(active_position_errors) ** 2)))
        if active_position_errors
        else None
    )
    maximum_task_position_rms = max(
        (
            values["position_rms_m"]
            for source_name, values in task_rms.items()
            if values["position_rms_m"] is not None
            and source_name not in {"left_calf", "right_calf"}
        ),
        default=None,
    )
    maximum_calf_task_position_rms = max(
        (
            task_rms[source_name]["position_rms_m"]
            for source_name in ("left_calf", "right_calf")
            if source_name in task_rms
            and task_rms[source_name]["position_rms_m"] is not None
        ),
        default=None,
    )
    available_ground_positions = [
        contact_body_positions[source_name]
        for source_name in expected_ground_references
        if source_name in contact_body_positions
    ]
    all_ground_heights = (
        np.concatenate([values[:, 2] for values in available_ground_positions])
        if available_ground_positions
        else np.empty(0, dtype=np.float64)
    )
    static_floor_p01 = (
        float(np.percentile(all_ground_heights, 1.0))
        if all_ground_heights.size
        else None
    )
    minimum_root_height = float(np.min(qpos[:, 2]))
    median_root_height = float(np.median(qpos[:, 2]))
    if actual_height_offset_mode == "static_sequence_floor":
        report.require(
            len(available_ground_positions) == len(expected_ground_references),
            "静态地板标定必须能 FK 得到全部四个机器人足点",
        )
        report.require(
            static_floor_p01 is not None
            and abs(static_floor_p01) < MAX_STATIC_FLOOR_P01_ERROR_M,
            "四足点全序列 1% 高度未对齐地面: "
            f"actual={static_floor_p01}, threshold={MAX_STATIC_FLOOR_P01_ERROR_M}",
        )
        report.require(
            minimum_root_height > MIN_STATIC_ROOT_HEIGHT_M,
            "机器人根节点仍整体陷入地面: "
            f"actual={minimum_root_height}, threshold={MIN_STATIC_ROOT_HEIGHT_M}",
        )
    report.require(
        aggregate_position_rms is not None
        and aggregate_position_rms < MAX_AGGREGATE_TASK_POSITION_RMS_M,
        f"有效任务整体位置 RMS 超阈值: {aggregate_position_rms}",
    )
    report.require(
        maximum_task_position_rms is not None
        and maximum_task_position_rms < MAX_SINGLE_TASK_POSITION_RMS_M,
        f"最差非 calf 单任务位置 RMS 超阈值: {maximum_task_position_rms}",
    )
    report.require(
        maximum_calf_task_position_rms is not None
        and maximum_calf_task_position_rms < MAX_CALF_TASK_POSITION_RMS_M,
        f"最差摆动期 calf 位置 RMS 超阈值: {maximum_calf_task_position_rms}",
    )

    paired_support_enabled = str(config.get("contact_task_mode")) == "paired_support"
    foot_slide = {}
    all_interval_displacements = []
    all_active_speeds = []
    maximum_penetration = 0.0
    for source_name, world_positions in contact_body_positions.items():
        state_idx = contact_names.index(source_name)
        states = contact_states[:, state_idx]
        if (
            str(config.get("contact_task_mode")) == "paired_support"
            and source_name.endswith("_toe")
        ):
            heel_name = source_name[: -len("_toe")] + "_foot_end"
            states = states | contact_states[:, contact_names.index(heel_name)]
        intervals = contact_intervals(states)
        interval_displacements = []
        interval_mean_speeds = []
        interval_p95_speeds = []
        for start, end in intervals:
            segment = world_positions[start:end]
            displacement = float(np.linalg.norm(segment[-1, :2] - segment[0, :2]))
            speeds = (
                np.linalg.norm(np.diff(segment[:, :2], axis=0), axis=1) * fps
                if len(segment) > 1
                else np.zeros(1)
            )
            interval_displacements.append(displacement)
            interval_mean_speeds.append(float(np.mean(speeds)))
            interval_p95_speeds.append(float(np.percentile(speeds, 95)))
            all_interval_displacements.append(displacement)
            all_active_speeds.extend(speeds.tolist())
        penetration = (
            float(max(0.0, -np.min(world_positions[states, 2])))
            if np.any(states)
            else 0.0
        )
        maximum_penetration = max(maximum_penetration, penetration)
        active_heights = world_positions[states, 2]
        foot_slide[source_name] = {
            "interval_count": len(intervals),
            "interval_displacement_m": interval_displacements,
            "interval_mean_horizontal_speed_mps": interval_mean_speeds,
            "interval_p95_horizontal_speed_mps": interval_p95_speeds,
            "maximum_penetration_m": penetration,
            "maximum_contact_height_m": float(np.max(active_heights)) if active_heights.size else None,
        }
    median_displacement = None
    median_speed = None
    if not all_interval_displacements:
        report.warnings.append("动作中没有可评估的脚部接触区间")
    else:
        median_displacement = float(np.median(all_interval_displacements))
        median_speed = float(np.median(all_active_speeds)) if all_active_speeds else 0.0
        if paired_support_enabled:
            report.require(median_displacement < 0.03, f"接触区间中位水平位移超阈值: {median_displacement}")
            report.require(median_speed < 0.05, f"接触期间中位水平速度超阈值: {median_speed}")
            report.require(maximum_penetration < 0.01, f"脚底最大穿透超阈值: {maximum_penetration}")
        else:
            # G1 式 legacy_hold 只用低权重接触帮助 IK，不承诺足底严格锁定。
            # 超过原 paired_support 门槛时仍写 warning，既不掩盖质量代价，也不让
            # 用户明确选择的连续性优先模式被贴地验收错误阻断。
            if median_displacement >= 0.03:
                report.warnings.append(
                    f"legacy_hold 接触区间中位水平位移较大: {median_displacement}"
                )
            if median_speed >= 0.05:
                report.warnings.append(
                    f"legacy_hold 接触期间中位水平速度较大: {median_speed}"
                )
            if maximum_penetration >= 0.01:
                report.warnings.append(
                    f"legacy_hold 脚底最大穿透较大: {maximum_penetration}"
                )

    # paired_support 才承诺绝对贴地；heel/toe 高度差软任务只承诺平足期尽量等高，
    # 不锁足点 XY 或绝对 z。两者分别使用自己的渐变窗口，避免验收把软任务误当成
    # 高权重支撑锁定。关节连续性、任务误差和数值合同仍统一验收。
    ground_clearance = float(config.get("post_ik_ground_clearance", 0.0))
    contact_ramp_frames = max(
        1,
        int(round(float(config.get("contact_weight_ramp_seconds", 0.0)) * fps)),
    )
    height_soft_enabled = float(
        config.get("heel_toe_height_difference_cost", 0.0)
    ) > 0.0
    height_ramp_frames = max(
        1,
        int(
            round(
                float(
                    config.get("heel_toe_height_difference_ramp_seconds", 0.0)
                )
                * fps
            )
        ),
    )

    def settled_support_mask(states: np.ndarray, ramp_frames: int) -> np.ndarray:
        """排除每个接触区间权重尚未升到 1 的前沿帧。"""
        settled = np.asarray(states, dtype=bool).copy()
        for start, end in contact_intervals(settled):
            settled[start : min(end, start + ramp_frames - 1)] = False
        return settled

    paired_support_quality = {}
    all_support_height_above_ground = []
    all_flat_height_differences = []
    for side in ("left", "right") if (paired_support_enabled or height_soft_enabled) else ():
        heel_name = f"{side}_foot_end"
        toe_name = f"{side}_toe"
        if heel_name not in contact_body_positions or toe_name not in contact_body_positions:
            report.failures.append(f"缺少 {side} heel/toe 成对支撑输出")
            continue
        heel_states = contact_states[:, contact_names.index(heel_name)]
        toe_states = contact_states[:, contact_names.index(toe_name)]
        flat_states = heel_states if paired_support_enabled else (heel_states & toe_states)
        toe_off_states = ~heel_states & toe_states
        support_states = heel_states | toe_states
        if bool(config.get("require_ground_contact_detection", False)):
            report.require(
                bool(np.any(support_states)),
                f"{side} 脚必须至少检测到一个 heel/toe 支撑帧",
            )
        settled_flat_states = settled_support_mask(
            flat_states,
            contact_ramp_frames if paired_support_enabled else height_ramp_frames,
        )
        settled_support_states = settled_support_mask(
            support_states, contact_ramp_frames
        )
        heel_z = contact_body_positions[heel_name][:, 2]
        toe_z = contact_body_positions[toe_name][:, 2]
        support_heights = np.concatenate(
            [heel_z[settled_flat_states], toe_z[settled_support_states]]
        )
        height_above_ground = support_heights - ground_clearance
        flat_height_difference = np.abs(
            toe_z[settled_flat_states] - heel_z[settled_flat_states]
        )
        all_phase_flat_height_difference = np.abs(
            toe_z[flat_states] - heel_z[flat_states]
        )
        all_support_height_above_ground.extend(height_above_ground.tolist())
        all_flat_height_differences.extend(flat_height_difference.tolist())
        paired_support_quality[side] = {
            "flat_frames": int(np.count_nonzero(flat_states)),
            "settled_flat_frames": int(np.count_nonzero(settled_flat_states)),
            "toe_off_frames": int(np.count_nonzero(toe_off_states)),
            "maximum_support_point_height_above_ground_m": (
                float(np.max(height_above_ground)) if height_above_ground.size else None
            ),
            "maximum_flat_heel_toe_height_difference_m": (
                float(np.max(flat_height_difference))
                if flat_height_difference.size
                else None
            ),
            "maximum_all_phase_flat_heel_toe_height_difference_m": (
                float(np.max(all_phase_flat_height_difference))
                if all_phase_flat_height_difference.size
                else None
            ),
        }
    maximum_support_height = (
        float(np.max(all_support_height_above_ground))
        if all_support_height_above_ground
        else None
    )
    maximum_flat_height_difference = (
        float(np.max(all_flat_height_differences))
        if all_flat_height_differences
        else None
    )
    if paired_support_enabled:
        report.require(
            maximum_support_height is not None
            and maximum_support_height < MAX_SUPPORT_POINT_HEIGHT_ABOVE_GROUND_M,
            f"支撑足点离地高度超阈值: {maximum_support_height}",
        )
        report.require(
            maximum_flat_height_difference is not None
            and maximum_flat_height_difference < MAX_FLAT_HEEL_TOE_HEIGHT_DIFFERENCE_M,
            f"平足期 heel/toe 高度差超阈值: {maximum_flat_height_difference}",
        )
    elif height_soft_enabled:
        if maximum_flat_height_difference is None:
            report.warnings.append("动作中没有可评估的 heel/toe 同时接触帧")
        elif maximum_flat_height_difference >= MAX_FLAT_HEEL_TOE_HEIGHT_DIFFERENCE_M:
            report.warnings.append(
                "heel/toe 等高软约束残差较大: "
                f"{maximum_flat_height_difference}"
            )

    ik_statistics = None
    if metadata_path is not None:
        metadata = json.loads(metadata_path.expanduser().resolve().read_text(encoding="utf-8"))
        report.require(int(metadata.get("num_frames", -1)) == frame_count, "元数据帧数不匹配")
        report.require(int(metadata.get("qpos_size", -1)) == model.nq, "元数据 qpos_size 不匹配")
        current_config_sha = sha256_file(config_path)
        report.require(
            metadata.get("config_sha256") == current_config_sha,
            "元数据配置 SHA256 与当前配置不一致: "
            f"expected={current_config_sha}, actual={metadata.get('config_sha256')}",
        )
        ik_statistics = metadata.get("ik_statistics")
        report.require(
            float(metadata.get("heel_toe_height_difference_cost", -1.0))
            == float(config.get("heel_toe_height_difference_cost", 0.0)),
            "元数据 heel/toe 高度差 cost 与配置不一致",
        )
        report.require(
            float(
                metadata.get("heel_toe_height_difference_ramp_seconds", -1.0)
            )
            == float(config.get("heel_toe_height_difference_ramp_seconds", 0.0)),
            "元数据 heel/toe 高度差渐变时间与配置不一致",
        )
    report.checks["motion"] = {
        "frames": frame_count,
        "fps": fps,
        "ground_reference_contacts": actual_ground_references,
        "contact_height_dynamic_offset_enabled": actual_dynamic_height_offset,
        "contact_height_offset_mode": actual_height_offset_mode,
        "contact_height_floor_method": actual_floor_method,
        "contact_height_source_floor_fit": source_floor_fit,
        "contact_height_retarget_floor_fit": retarget_floor_fit,
        "contact_height_offset_min_m": height_offset_min,
        "contact_height_offset_median_m": height_offset_median,
        "contact_height_offset_max_m": height_offset_max,
        "csv_shape": list(qpos.shape),
        "nan_inf_count": finite_count,
        "root_quaternion_norm_max_error": root_norm_error,
        "joint_limit_violation_count": violation_count,
        "joint_limit_near_rate": joint_limit_near_rate,
        "maximum_single_joint_exact_limit_rate": maximum_exact_limit_rate,
        "joint_limit_occupancy_validation": joint_limit_occupancy_validation,
        "per_joint_limits": per_joint_limits,
        "max_abs_joint_velocity_rad_s": maximum_joint_velocity,
        "p99_abs_joint_acceleration_rad_s2": p99_joint_acceleration,
        "max_abs_joint_acceleration_rad_s2": maximum_joint_acceleration,
        "p99_abs_joint_jerk_rad_s3": p99_joint_jerk,
        "max_abs_joint_jerk_rad_s3": maximum_joint_jerk,
        "aggregate_task_position_rms_m": aggregate_position_rms,
        "maximum_task_position_rms_m": maximum_task_position_rms,
        "maximum_calf_task_position_rms_m": maximum_calf_task_position_rms,
        "global_ground_alignment": {
            "mode": actual_height_offset_mode,
            "reference_contacts": expected_ground_references,
            "foot_height_p01_m": static_floor_p01,
            "foot_height_min_m": (
                float(np.min(all_ground_heights))
                if all_ground_heights.size
                else None
            ),
            "foot_height_median_m": (
                float(np.median(all_ground_heights))
                if all_ground_heights.size
                else None
            ),
            "minimum_root_height_m": minimum_root_height,
            "median_root_height_m": median_root_height,
        },
        "quality_thresholds": {
            "aggregate_task_position_rms_m": MAX_AGGREGATE_TASK_POSITION_RMS_M,
            "single_task_position_rms_m": MAX_SINGLE_TASK_POSITION_RMS_M,
            "calf_task_position_rms_m": MAX_CALF_TASK_POSITION_RMS_M,
            "joint_limit_near_rate": MAX_JOINT_LIMIT_NEAR_RATE,
            "single_joint_exact_limit_rate": MAX_SINGLE_JOINT_EXACT_LIMIT_RATE,
            "frame_to_frame_joint_velocity_rad_s": (
                MAX_FRAME_TO_FRAME_JOINT_VELOCITY_RAD_S
            ),
            "p99_joint_acceleration_rad_s2": MAX_P99_JOINT_ACCELERATION_RAD_S2,
            "max_joint_acceleration_rad_s2": MAX_ABS_JOINT_ACCELERATION_RAD_S2,
            "p99_joint_jerk_rad_s3": MAX_P99_JOINT_JERK_RAD_S3,
            "max_joint_jerk_rad_s3": MAX_ABS_JOINT_JERK_RAD_S3,
            "support_point_height_above_ground_m": (
                MAX_SUPPORT_POINT_HEIGHT_ABOVE_GROUND_M
            ),
            "flat_heel_toe_height_difference_m": (
                MAX_FLAT_HEEL_TOE_HEIGHT_DIFFERENCE_M
            ),
            "static_floor_p01_error_m": MAX_STATIC_FLOOR_P01_ERROR_M,
            "minimum_static_root_height_m": MIN_STATIC_ROOT_HEIGHT_M,
        },
        "task_rms": task_rms,
        "foot_slide": foot_slide,
        "foot_slide_summary": {
            "median_interval_displacement_m": median_displacement,
            "median_active_horizontal_speed_mps": median_speed,
            "maximum_penetration_m": maximum_penetration,
        },
        "paired_support_quality": paired_support_quality,
        "paired_support_summary": {
            "enabled": paired_support_enabled,
            "contact_weight_ramp_frames": contact_ramp_frames,
            "maximum_support_point_height_above_ground_m": maximum_support_height,
            "maximum_flat_heel_toe_height_difference_m": (
                maximum_flat_height_difference
            ),
        },
        "heel_toe_height_soft_summary": {
            "enabled": height_soft_enabled,
            "cost": float(config.get("heel_toe_height_difference_cost", 0.0)),
            "ramp_frames": height_ramp_frames,
            "quality": paired_support_quality,
            "maximum_settled_height_difference_m": (
                maximum_flat_height_difference
            ),
        },
        "ik_statistics": ik_statistics,
    }
    return frame_count


def validate_npz(
    npz_path: Path,
    frame_count: int,
    expected_fps: float,
    report: ValidationReport,
) -> None:
    """严格验证 IsaacLab Mimic NPZ 的字段、shape、帧率和四元数。"""
    resolved = npz_path.expanduser().resolve()
    with np.load(resolved) as payload:
        expected_shapes = {
            "joint_pos": (frame_count, 21),
            "joint_vel": (frame_count, 21),
            "body_pos_w": (frame_count, 22, 3),
            "body_quat_w": (frame_count, 22, 4),
            "body_lin_vel_w": (frame_count, 22, 3),
            "body_ang_vel_w": (frame_count, 22, 3),
        }
        actual_shapes = {}
        for field_name, expected in expected_shapes.items():
            report.require(field_name in payload, f"NPZ 缺少字段: {field_name}")
            if field_name not in payload:
                continue
            values = payload[field_name]
            actual_shapes[field_name] = list(values.shape)
            report.require(values.shape == expected, f"NPZ {field_name} shape 错误: expected={expected}, actual={values.shape}")
            report.require(np.all(np.isfinite(values)), f"NPZ {field_name} 包含 NaN/Inf")
        fps = float(payload["fps"])
        report.require(
            abs(fps - expected_fps) < 1.0e-9,
            f"NPZ fps 与配置不一致: expected={expected_fps}, actual={fps}",
        )
        if "body_quat_w" in payload:
            norm_error = float(
                np.max(np.abs(np.linalg.norm(payload["body_quat_w"], axis=-1) - 1.0))
            )
            report.require(norm_error < 1.0e-5, f"NPZ body 四元数范数误差过大: {norm_error}")
        else:
            norm_error = None
        anchor = str(payload["anchor_body_name"].item())
        order = str(payload["quaternion_order"].item())
        report.require(anchor == "waist_yaw_link", f"anchor_body_name 错误: {anchor}")
        report.require(order == "wxyz", f"quaternion_order 错误: {order}")
        report.require(payload["joint_names"].tolist() == BUMI3_JOINT_NAMES, "NPZ 关节顺序错误")
        report.require(payload["body_names"].tolist() == BUMI3_ISAAC_BODY_NAMES, "NPZ body 顺序错误")
    report.checks["mimic_npz"] = {
        "path": str(resolved),
        "fps": fps,
        "shapes": actual_shapes,
        "body_quaternion_norm_max_error": norm_error,
        "anchor_body_name": anchor,
        "quaternion_order": order,
    }


def main() -> None:
    args = parse_args()
    report = ValidationReport()
    config_path = args.config.expanduser().resolve()
    try:
        config = load_yaml(config_path)
        model, _ = validate_model(config, config_path, report)
        artifact_args = [args.keypoints, args.csv, args.npz]
        if any(value is not None for value in artifact_args):
            report.require(all(value is not None for value in artifact_args), "完整验证必须同时传入 --keypoints、--csv 和 --npz")
            if all(value is not None for value in artifact_args):
                frame_count = validate_motion(
                    model,
                    config,
                    config_path,
                    args.keypoints,
                    args.csv,
                    args.metadata,
                    report,
                )
                expected_fps = float(config["output"]["target_fps"])
                validate_npz(args.npz, frame_count, expected_fps, report)
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError) as error:
        report.failures.append(str(error))

    output = args.report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = report.payload()
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "status": payload["status"], "failures": payload["failures"]}, ensure_ascii=False, indent=2))
    if report.failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
