#!/usr/bin/env python3
"""BUMI3 重定向流水线共享的数据契约与数值工具。

本文件集中定义 BUMI3 在 MuJoCo、IsaacLab Mimic 与本仓库之间必须保持一致的
21 个关节顺序、22 个物理 body 顺序、body 名称别名和四元数约定。导出、验证、
播放等脚本都从这里读取同一份契约，避免各脚本复制列表后发生静默漂移。文件还
提供按关节名查询 MuJoCo qpos 地址、SHA256、四元数连续化和有限差分等小工具；
这些工具只处理格式与运动学数据，不改变仓库既有的单阶段 Mink IK 主干。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


BUMI3_JOINT_NAMES = [
    "l_leg_pitch_joint",
    "r_leg_pitch_joint",
    "waist_yaw_joint",
    "l_leg_roll_joint",
    "r_leg_roll_joint",
    "l_arm_pitch_joint",
    "r_arm_pitch_joint",
    "l_leg_yaw_joint",
    "r_leg_yaw_joint",
    "l_arm_roll_joint",
    "r_arm_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
]

BUMI3_ISAAC_BODY_NAMES = [
    "base_link",
    "waist_yaw_link",
    "l_arm_pitch_link",
    "l_arm_roll_link",
    "l_arm_yaw_link",
    "l_elbow_pitch_link",
    "r_arm_pitch_link",
    "r_arm_roll_link",
    "r_arm_yaw_link",
    "r_elbow_pitch_link",
    "l_leg_pitch_link",
    "l_leg_roll_link",
    "l_leg_yaw_link",
    "l_knee_pitch_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "r_leg_pitch_link",
    "r_leg_roll_link",
    "r_leg_yaw_link",
    "r_knee_pitch_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
]

BUMI3_MARKER_BODY_NAMES = [
    "hips_sphere",
    "neck_sphere",
    "head_sphere",
    "left_foot_end_link",
    "left_toe_link",
    "right_foot_end_link",
    "right_toe_link",
    "left_hand",
    "right_hand",
]


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 并确保顶层为映射。"""
    if not path.is_file():
        raise FileNotFoundError(f"YAML 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML 顶层必须是映射: path={path}, actual={type(value).__name__}")
    return value


def resolve_config_path(config_path: Path, value: str) -> Path:
    """将配置中的仓库相对路径解析为绝对路径。"""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    repository_root = config_path.expanduser().resolve().parents[2]
    return (repository_root / path).resolve()


def sha256_file(path: Path) -> str:
    """计算文件 SHA256，供资产与输出元数据审计。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def joint_qpos_addresses(model: mujoco.MjModel, joint_names: list[str]) -> dict[str, int]:
    """按名称返回单自由度关节 qpos 地址，并拒绝缺失或多自由度关节。"""
    result: dict[str, int] = {}
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo 模型缺少关节: field=joint_names, expected={name}")
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(
                f"目标关节必须是单自由度: joint={name}, actual_type={joint_type}"
            )
        result[name] = int(model.jnt_qposadr[joint_id])
    return result


def normalize_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """归一化 wxyz 四元数并检查零范数。"""
    values = np.asarray(quaternions, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError(f"wxyz 四元数末维必须为 4: actual={values.shape}")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("wxyz 四元数包含零范数")
    return values / norms


def make_quaternions_continuous_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """逐时间轴统一四元数半球，消除 q 与 -q 的符号跳变。"""
    values = normalize_quaternions_wxyz(quaternions).copy()
    if values.ndim < 2:
        raise ValueError(f"四元数序列至少需要时间维和分量维: actual={values.shape}")
    for frame_idx in range(1, values.shape[0]):
        dot = np.sum(values[frame_idx - 1] * values[frame_idx], axis=-1, keepdims=True)
        values[frame_idx] = np.where(dot < 0.0, -values[frame_idx], values[frame_idx])
    return values


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    """以首尾单边、内部中心差分计算任意向量序列速度。"""
    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] < 2:
        return np.zeros_like(array)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps 必须为正有限值: actual={fps}")
    dt = 1.0 / float(fps)
    result = np.empty_like(array)
    result[0] = (array[1] - array[0]) / dt
    result[-1] = (array[-1] - array[-2]) / dt
    if array.shape[0] > 2:
        result[1:-1] = (array[2:] - array[:-2]) / (2.0 * dt)
    return result


def quaternion_angular_velocity_wxyz(quaternions: np.ndarray, fps: float) -> np.ndarray:
    """用相对旋转对数计算 wxyz body 四元数序列的世界角速度。"""
    values = make_quaternions_continuous_wxyz(quaternions)
    if values.ndim != 3 or values.shape[-1] != 4:
        raise ValueError(f"body 四元数必须为 [T,B,4]: actual={values.shape}")
    if values.shape[0] < 2:
        return np.zeros(values.shape[:-1] + (3,), dtype=np.float64)
    dt = 1.0 / float(fps)
    rotations = Rotation.from_quat(values[..., [1, 2, 3, 0]].reshape(-1, 4))
    matrices = rotations.as_matrix().reshape(values.shape[0], values.shape[1], 3, 3)
    result = np.zeros(values.shape[:-1] + (3,), dtype=np.float64)

    def relative_rotvec(start: int, end: int) -> np.ndarray:
        relative = np.einsum(
            "...ji,...jk->...ik", matrices[start], matrices[end]
        )
        return Rotation.from_matrix(relative).as_rotvec()

    result[0] = relative_rotvec(0, 1) / dt
    result[-1] = relative_rotvec(values.shape[0] - 2, values.shape[0] - 1) / dt
    for frame_idx in range(1, values.shape[0] - 1):
        result[frame_idx] = relative_rotvec(frame_idx - 1, frame_idx + 1) / (2.0 * dt)
    return result
