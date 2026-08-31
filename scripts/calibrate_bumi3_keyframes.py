#!/usr/bin/env python3
"""用独立中立参考姿态标定 BUMI3 关键姿态坐标轴。

脚本先构造一帧 betas/pose/root_orient 全零的中性 SMPL-X，并通过
``scripts/smpl_replay.py`` 的正式连杆缩放逻辑得到 BUMI3 尺寸的位置目标。随后固定
源坐标到 BUMI3 坐标所需的根节点 yaw，使用左右严格对称的 11 参数人体姿态拟合
髋、踝、肩、肘和手的位置；膝关节位置不参与拟合，避免 SMPL 髋宽与机器人髋宽
不同导致双腿外撇。优化会遵守正式 IK 收紧后的关节限位，并用正则项保持手肘接近
伸直、关节姿态自然。

最后，脚本在这个“位置与姿态一致”的机器人参考姿态上读取各目标 body 世界旋转，
计算 ``R_source.T @ R_robot``，按仓库的右乘约定生成每个关键帧的轴映射。标定参考
姿态与正式 IK 热启动姿态是两个不同合同：``reference_pose.fixed_parameters`` 可把
标定脚踝固定为零角，保证中立脚掌水平；``--write`` 只更新根坐标朝向、标定结果快照
和 ``key_frame_config``，绝不再覆盖 ``initial_joint_positions``。这样热启动可以保持
易收敛的屈膝姿态，而方向标定不会继承热启动脚掌的预倾角。
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from bumi3_common import joint_qpos_addresses, load_yaml, resolve_config_path
from smpl_replay import (
    REPLAY_BODY_NAMES,
    build_replay_buffers,
    build_retarget_keypoints,
    compute_robot_link_lengths,
)


SOURCE_TO_ROBOT_BODY = {
    "hips_mean": "hips_sphere",
    "left_up_leg": "l_leg_pitch_link",
    "left_leg": "l_knee_pitch_link",
    "left_foot": "l_ankle_roll_link",
    "right_up_leg": "r_leg_pitch_link",
    "right_leg": "r_knee_pitch_link",
    "right_foot": "r_ankle_roll_link",
    "shoulder_mean": "neck_sphere",
    "left_arm": "l_arm_pitch_link",
    "left_fore_arm": "l_elbow_pitch_link",
    "left_hand": "left_hand",
    "right_arm": "r_arm_pitch_link",
    "right_fore_arm": "r_elbow_pitch_link",
    "right_hand": "right_hand",
    "head": "head_sphere",
}


# BUMI3 左右两侧的关节拓扑完全镜像。用共享参数显式保持对称，比在 21 个独立
# 关节上加一个软惩罚更可靠，也能阻止数值优化用左右不同的肘弯曲拟合 SMPL 的
# 毫米级不对称。
SYMMETRIC_REFERENCE_PARAMETERS = {
    "waist_yaw": (("waist_yaw_joint", 1.0),),
    "arm_pitch": (("l_arm_pitch_joint", 1.0), ("r_arm_pitch_joint", 1.0)),
    "arm_roll": (("l_arm_roll_joint", 1.0), ("r_arm_roll_joint", -1.0)),
    "arm_yaw": (("l_arm_yaw_joint", 1.0), ("r_arm_yaw_joint", -1.0)),
    "elbow_pitch": (("l_elbow_pitch_joint", 1.0), ("r_elbow_pitch_joint", 1.0)),
    "leg_pitch": (("l_leg_pitch_joint", 1.0), ("r_leg_pitch_joint", 1.0)),
    "leg_roll": (("l_leg_roll_joint", 1.0), ("r_leg_roll_joint", -1.0)),
    "leg_yaw": (("l_leg_yaw_joint", 1.0), ("r_leg_yaw_joint", -1.0)),
    "knee_pitch": (("l_knee_pitch_joint", 1.0), ("r_knee_pitch_joint", 1.0)),
    "ankle_pitch": (("l_ankle_pitch_joint", 1.0), ("r_ankle_pitch_joint", 1.0)),
    "ankle_roll": (("l_ankle_roll_joint", 1.0), ("r_ankle_roll_joint", -1.0)),
}

MIRRORED_SOURCE_PAIRS = (
    ("left_hip", "right_hip"),
    ("left_thigh", "right_thigh"),
    ("left_calf", "right_calf"),
    ("left_shoulder", "right_shoulder"),
    ("left_arm", "right_arm"),
    ("left_fore_arm", "right_fore_arm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动标定 BUMI3 key_frame_config")
    parser.add_argument("--robot-config", type=Path, default=Path("config/robot/bumi3.yaml"))
    parser.add_argument("--smpl-model-path", type=Path, required=True)
    parser.add_argument("--model-type", choices=["smpl", "smplx"], default="smplx")
    parser.add_argument("--gender", choices=["neutral", "female", "male"], default="neutral")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def _neutral_source_pose(
    model_path: Path, model_type: str, gender: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with tempfile.NamedTemporaryFile(suffix=".npz") as file:
        np.savez(
            file.name,
            surface_model_type=np.asarray(model_type),
            trans=np.zeros((1, 3), dtype=np.float32),
            root_orient=np.zeros((1, 3), dtype=np.float32),
            pose_body=np.zeros((1, 63), dtype=np.float32),
            betas=np.zeros(10, dtype=np.float32),
            gender=np.asarray(gender),
            mocap_frame_rate=np.asarray(30.0, dtype=np.float32),
        )
        positions, quaternions, _fps, _gender, _metadata = build_replay_buffers(
            motion_file=Path(file.name),
            smpl_model_path=model_path,
            gender_override=gender,
            device="cpu",
            chunk_size=1,
            translation_offset=np.zeros(3, dtype=np.float32),
        )
    return (
        {name: positions[0, index] for index, name in enumerate(REPLAY_BODY_NAMES)},
        {name: quaternions[0, index] for index, name in enumerate(REPLAY_BODY_NAMES)},
    )


def _free_joint_qpos_address(model: mujoco.MjModel) -> int:
    free_joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if free_joint_ids.size != 1:
        raise ValueError(f"BUMI3 必须恰好一个 freejoint: actual={free_joint_ids.size}")
    return int(model.jnt_qposadr[int(free_joint_ids[0])])


def _tightened_joint_bounds(
    model: mujoco.MjModel,
    config: dict[str, Any],
    joint_name: str,
) -> tuple[float, float]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"BUMI3 模型缺少参考姿态关节: {joint_name}")
    lower, upper = (float(value) for value in model.jnt_range[joint_id])
    for pattern, offsets in config.get("joints_limit_offset_degrees", {}).items():
        if str(pattern) not in joint_name:
            continue
        if isinstance(offsets, (list, tuple)):
            if len(offsets) != 2:
                raise ValueError(f"关节限位偏置必须为 [lower, upper]: {pattern}={offsets}")
            lower_offset, upper_offset = offsets
        else:
            lower_offset, upper_offset = offsets, 0.0
        lower += float(np.deg2rad(float(lower_offset)))
        upper += float(np.deg2rad(float(upper_offset)))
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


def _symmetrize_relative_targets(
    target_positions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    hips = np.asarray(target_positions["hips_mean"], dtype=np.float64)
    relative = {
        name: np.asarray(position, dtype=np.float64) - hips
        for name, position in target_positions.items()
    }
    mirror = np.asarray([-1.0, 1.0, 1.0], dtype=np.float64)
    for left_name, right_name in MIRRORED_SOURCE_PAIRS:
        symmetric_left = 0.5 * (relative[left_name] + mirror * relative[right_name])
        relative[left_name] = symmetric_left
        relative[right_name] = mirror * symmetric_left
    for center_name in ("hips_mean", "neck", "head"):
        if center_name in relative:
            relative[center_name][0] = 0.0
    return relative


def _solve_reference_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: dict[str, Any],
    target_positions: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    reference_config = config.get("reference_pose", {})
    root_yaw_degrees = float(reference_config.get("root_yaw_degrees", -90.0))
    regularization_weight = float(reference_config.get("regularization_weight", 0.1))
    if not np.isfinite(regularization_weight) or regularization_weight <= 0.0:
        raise ValueError(
            f"reference_pose.regularization_weight 必须为正有限值: {regularization_weight}"
        )
    flat_foot_orientation_weight = float(
        reference_config.get("flat_foot_orientation_weight", 0.0)
    )
    if (
        not np.isfinite(flat_foot_orientation_weight)
        or flat_foot_orientation_weight < 0.0
    ):
        raise ValueError(
            "reference_pose.flat_foot_orientation_weight 必须为非负有限值: "
            f"{flat_foot_orientation_weight}"
        )
    joint_seeds = {
        str(name): float(value)
        for name, value in reference_config.get("joint_seeds", {}).items()
    }
    fixed_parameters = {
        str(name): float(value)
        for name, value in reference_config.get("fixed_parameters", {}).items()
    }
    unknown_fixed = sorted(set(fixed_parameters) - set(SYMMETRIC_REFERENCE_PARAMETERS))
    if unknown_fixed:
        raise ValueError(f"reference_pose.fixed_parameters 含未知参数: {unknown_fixed}")
    if not all(np.isfinite(value) for value in fixed_parameters.values()):
        raise ValueError(
            f"reference_pose.fixed_parameters 必须全部为有限值: {fixed_parameters}"
        )
    if flat_foot_orientation_weight > 0.0 and fixed_parameters.get(
        "ankle_pitch"
    ) != 0.0:
        raise ValueError(
            "启用中立平足标定时必须显式固定 reference_pose ankle_pitch=0"
        )
    initial_joints = {
        str(name): float(value)
        for name, value in config.get("initial_joint_positions", {}).items()
    }
    joint_addresses = joint_qpos_addresses(
        model,
        [joint_name for pairs in SYMMETRIC_REFERENCE_PARAMETERS.values() for joint_name, _ in pairs],
    )

    parameter_names = [
        name for name in SYMMETRIC_REFERENCE_PARAMETERS if name not in fixed_parameters
    ]
    lower_bounds = []
    upper_bounds = []
    initial_values = []
    for parameter_name, joint_specs in SYMMETRIC_REFERENCE_PARAMETERS.items():
        candidate_values = []
        candidate_lowers = []
        candidate_uppers = []
        for joint_name, sign in joint_specs:
            lower, upper = _tightened_joint_bounds(model, config, joint_name)
            if sign < 0.0:
                lower, upper = -upper, -lower
            candidate_lowers.append(lower)
            candidate_uppers.append(upper)
            seed = joint_seeds.get(joint_name, initial_joints.get(joint_name, 0.0))
            candidate_values.append(sign * seed)
        lower = max(candidate_lowers)
        upper = min(candidate_uppers)
        if lower > upper:
            raise ValueError(
                f"左右对称关节没有共同可行域: parameter={parameter_name}, bounds=[{lower}, {upper}]"
            )
        initial = float(np.clip(np.mean(candidate_values), lower, upper))
        if parameter_name in fixed_parameters:
            fixed_value = fixed_parameters[parameter_name]
            if fixed_value < lower - 1.0e-12 or fixed_value > upper + 1.0e-12:
                raise ValueError(
                    "标定固定参数超出实际 IK 限位: "
                    f"parameter={parameter_name}, value={fixed_value}, bounds=[{lower}, {upper}]"
                )
            continue
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        initial_values.append(initial)

    root_address = _free_joint_qpos_address(model)
    reference_qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
    root_position = np.asarray(
        config.get("initial_root_pose", {}).get(
            "position", reference_qpos[root_address : root_address + 3]
        ),
        dtype=np.float64,
    )
    if root_position.shape != (3,) or not np.all(np.isfinite(root_position)):
        raise ValueError(f"BUMI3 初始根位置必须为 3 个有限值: {root_position}")
    reference_qpos[root_address : root_address + 3] = root_position
    reference_qpos[root_address + 3 : root_address + 7] = Rotation.from_euler(
        "z", root_yaw_degrees, degrees=True
    ).as_quat()[[3, 0, 1, 2]]

    relative_targets = _symmetrize_relative_targets(target_positions)
    ik_entries = []
    for source_name, entry in config.get("ik_match_table", {}).items():
        robot_body, position_cost, _rotation_cost = entry
        if float(position_cost) <= 0.0:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(robot_body))
        if body_id < 0:
            raise ValueError(f"BUMI3 模型缺少参考姿态 body: {robot_body}")
        ik_entries.append((str(source_name), int(body_id), float(position_cost)))
    hips_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, str(config["ik_match_table"]["hips_mean"][0])
    )
    foot_body_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            SOURCE_TO_ROBOT_BODY[f"{side}_foot"],
        )
        for side in ("left", "right")
    ]

    resolved_parameters: dict[str, float] = {}

    def apply_parameters(values: np.ndarray) -> None:
        resolved_parameters.clear()
        resolved_parameters.update(fixed_parameters)
        resolved_parameters.update(
            {name: float(value) for name, value in zip(parameter_names, values)}
        )
        for parameter_name, value in resolved_parameters.items():
            for joint_name, sign in SYMMETRIC_REFERENCE_PARAMETERS[parameter_name]:
                reference_qpos[joint_addresses[joint_name]] = sign * float(value)
        data.qpos[:] = reference_qpos
        mujoco.mj_forward(model, data)

    initial = np.asarray(initial_values, dtype=np.float64)

    def residual(values: np.ndarray) -> np.ndarray:
        apply_parameters(values)
        robot_hips = data.xpos[hips_body_id]
        position_residuals = []
        for source_name, body_id, position_cost in ik_entries:
            delta = (data.xpos[body_id] - robot_hips) - relative_targets[source_name]
            position_residuals.extend(np.sqrt(position_cost) * delta)
        flat_foot_residuals = []
        if flat_foot_orientation_weight > 0.0:
            # 足底 marker 位于 ankle-roll link 的局部 XY 平面。要求该 body 的
            # 局部 X/Y 轴世界 z 分量为零，即完整足底平面而非仅前后两点水平。
            scale = np.sqrt(flat_foot_orientation_weight)
            for body_id in foot_body_ids:
                rotation = data.xmat[body_id].reshape(3, 3)
                flat_foot_residuals.extend(scale * rotation[2, :2])
        regularization = regularization_weight * (values - initial)
        return np.concatenate(
            (
                np.asarray(position_residuals, dtype=np.float64),
                np.asarray(flat_foot_residuals, dtype=np.float64),
                regularization,
            )
        )

    solution = least_squares(
        residual,
        initial,
        bounds=(np.asarray(lower_bounds), np.asarray(upper_bounds)),
        max_nfev=2000,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
    )
    if not solution.success or not np.all(np.isfinite(solution.x)):
        raise RuntimeError(
            f"BUMI3 中立参考姿态优化失败: success={solution.success}, message={solution.message}"
        )
    apply_parameters(solution.x)
    unweighted_errors = [
        np.linalg.norm(
            (data.xpos[body_id] - data.xpos[hips_body_id]) - relative_targets[source_name]
        )
        for source_name, body_id, _position_cost in ik_entries
        if source_name != "hips_mean"
    ]
    diagnostics = {
        "root_yaw_degrees": root_yaw_degrees,
        "position_rms_m": float(np.sqrt(np.mean(np.square(unweighted_errors)))),
        "maximum_position_error_m": float(np.max(unweighted_errors)),
        "optimizer_cost": float(solution.cost),
        "optimizer_evaluations": int(solution.nfev),
        "parameters": {
            name: float(resolved_parameters[name])
            for name in SYMMETRIC_REFERENCE_PARAMETERS
        },
        "fixed_parameters": dict(fixed_parameters),
        "flat_foot_orientation_weight": flat_foot_orientation_weight,
        "flat_foot_world_z_components": {
            side: data.xmat[body_id].reshape(3, 3)[2, :2].astype(float).tolist()
            for side, body_id in zip(("left", "right"), foot_body_ids)
        },
    }
    return reference_qpos.copy(), diagnostics


def calibrate(
    config_path: Path,
    model_path: Path,
    model_type: str,
    gender: str,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    config_path = config_path.expanduser().resolve()
    config = load_yaml(config_path)
    robot_xml = resolve_config_path(config_path, str(config["robot_xml_path"]))
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    source_positions, source_quaternions = _neutral_source_pose(
        model_path.expanduser().resolve(), model_type, gender
    )
    robot_link_lengths = compute_robot_link_lengths(config_path, robot_xml)
    skeleton_config = config_path.parent.parent / "skeleton" / "skeleton.yaml"
    replay_positions = np.stack(
        [source_positions[name] for name in REPLAY_BODY_NAMES], axis=0
    )[None, ...]
    replay_quaternions = np.stack(
        [source_quaternions[name] for name in REPLAY_BODY_NAMES], axis=0
    )[None, ...]
    retarget_positions, _retarget_quaternions, *_ = build_retarget_keypoints(
        replay_positions,
        replay_quaternions,
        robot_link_lengths,
        config_path,
        skeleton_config,
    )
    target_names = ["hips_mean", *config["robot_links"].keys()]
    target_positions = {
        name: retarget_positions[0, index]
        for index, name in enumerate(target_names)
    }
    reference_qpos, reference_diagnostics = _solve_reference_pose(
        model, data, config, target_positions
    )
    result: dict[str, Any] = {}
    for source_name, robot_body_name in SOURCE_TO_ROBOT_BODY.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body_name)
        if body_id < 0:
            raise ValueError(f"BUMI3 模型缺少标定 body: {robot_body_name}")
        source_quat = source_quaternions[source_name]
        source_rotation = Rotation.from_quat(source_quat[[1, 2, 3, 0]]).as_matrix()
        robot_rotation = data.xmat[body_id].reshape(3, 3)
        axis_map = source_rotation.T @ robot_rotation
        orthogonality = float(np.max(np.abs(axis_map.T @ axis_map - np.eye(3))))
        determinant = float(np.linalg.det(axis_map))
        residual = Rotation.from_matrix((source_rotation @ axis_map).T @ robot_rotation).magnitude()
        if orthogonality > 1.0e-6 or abs(determinant - 1.0) > 1.0e-6 or residual > 1.0e-4:
            raise ValueError(
                f"坐标轴标定失败: source={source_name}, body={robot_body_name}, "
                f"orthogonality={orthogonality}, determinant={determinant}, residual={residual}"
            )
        result[source_name] = {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [float(value) for value in axis_map[:, 0]],
                "y": [float(value) for value in axis_map[:, 1]],
                "z": [float(value) for value in axis_map[:, 2]],
            },
            "calibration": {
                "robot_body": robot_body_name,
                "neutral_residual_rad": float(residual),
            },
        }
    return result, reference_qpos, reference_diagnostics


def main() -> None:
    args = parse_args()
    result, reference_qpos, reference_diagnostics = calibrate(
        args.robot_config, args.smpl_model_path, args.model_type, args.gender
    )
    printable = {name: {key: value for key, value in entry.items() if key != "calibration"} for name, entry in result.items()}
    if args.write:
        config_path = args.robot_config.expanduser().resolve()
        config = load_yaml(config_path)
        robot_xml = resolve_config_path(config_path, str(config["robot_xml_path"]))
        model = mujoco.MjModel.from_xml_path(str(robot_xml))
        root_address = _free_joint_qpos_address(model)
        config.setdefault("initial_root_pose", {})["quaternion_wxyz"] = [
            float(value) for value in reference_qpos[root_address + 3 : root_address + 7]
        ]
        solved_joint_positions = config.setdefault("reference_pose", {}).setdefault(
            "solved_joint_positions", {}
        )
        for joint_name, address in joint_qpos_addresses(
            model,
            [joint_name for pairs in SYMMETRIC_REFERENCE_PARAMETERS.values() for joint_name, _ in pairs],
        ).items():
            solved_joint_positions[joint_name] = float(reference_qpos[address])
        config["key_frame_config"] = printable
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=120),
            encoding="utf-8",
        )
        print(f"已写入 BUMI3 key_frame_config: {config_path}")
    print(
        "reference_pose: "
        f"root_yaw_deg={reference_diagnostics['root_yaw_degrees']:.3f}, "
        f"position_rms_m={reference_diagnostics['position_rms_m']:.6f}, "
        f"max_position_error_m={reference_diagnostics['maximum_position_error_m']:.6f}, "
        f"optimizer_cost={reference_diagnostics['optimizer_cost']:.6f}, "
        f"evaluations={reference_diagnostics['optimizer_evaluations']}"
    )
    for name, value in reference_diagnostics["parameters"].items():
        print(f"reference_parameter.{name}={value:.9f}")
    for name, entry in result.items():
        print(
            f"{name}: body={entry['calibration']['robot_body']}, "
            f"neutral_residual_rad={entry['calibration']['neutral_residual_rad']:.3e}"
        )


if __name__ == "__main__":
    main()
