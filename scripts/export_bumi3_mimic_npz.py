#!/usr/bin/env python3
"""将 BUMI3 MuJoCo qpos CSV 导出为 IsaacLab Mimic 可直接读取的 NPZ。

输入 CSV 延续仓库既有契约：前三列为世界根位置，随后四列为 ``xyzw`` 根
四元数，剩余列为 MuJoCo qpos。脚本先借助同名 JSON 元数据核对帧率、qpos
维度和资产身份，再把根四元数转换回 MuJoCo 的 ``wxyz``，逐帧执行
``mj_forward``。关节位置通过关节名及 qpos 地址按 IsaacLab 的 21 关节顺序
重排；22 个物理 body 的世界位姿也严格按配置顺序提取。线速度与关节速度
使用首尾单边、内部中心差分，body 角速度使用连续化四元数的相对旋转对数，
因此不会把 ``q`` 与 ``-q`` 的等价符号变化误判为角速度尖峰。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from bumi3_common import (
    BUMI3_ISAAC_BODY_NAMES,
    BUMI3_JOINT_NAMES,
    finite_difference,
    joint_qpos_addresses,
    load_yaml,
    make_quaternions_continuous_wxyz,
    quaternion_angular_velocity_wxyz,
    resolve_config_path,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 BUMI3 IsaacLab Mimic NPZ")
    parser.add_argument("--csv", type=Path, required=True, help="BUMI3 qpos CSV")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV 同名元数据 JSON")
    parser.add_argument(
        "--config", type=Path, default=Path("config/robot/bumi3.yaml"), help="BUMI3 配置"
    )
    parser.add_argument("--output", type=Path, required=True, help="输出 NPZ")
    return parser.parse_args()


def load_csv_qpos(csv_path: Path, model: mujoco.MjModel) -> np.ndarray:
    """读取 CSV 并把根四元数从 xyzw 转换为单位 wxyz。"""
    resolved = csv_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"CSV 不存在: path={resolved}")
    values = np.loadtxt(resolved, delimiter=",", dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != model.nq:
        raise ValueError(
            f"CSV qpos shape 错误: path={resolved}, expected=[T,{model.nq}], actual={values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"CSV 包含 NaN/Inf: path={resolved}")
    qpos = values.copy()
    xyzw = values[:, 3:7]
    qpos[:, 3:7] = xyzw[:, [3, 0, 1, 2]]
    norms = np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError(f"CSV 根四元数包含零范数: path={resolved}")
    qpos[:, 3:7] /= norms
    return qpos


def resolve_body_names(
    model: mujoco.MjModel,
    isaac_body_names: list[str],
    aliases: dict[str, str],
) -> list[str]:
    """兼容 alias 的两个书写方向，并返回实际 MuJoCo body 名。"""
    reverse_aliases = {str(value): str(key) for key, value in aliases.items()}
    resolved = []
    for isaac_name in isaac_body_names:
        candidates = [isaac_name]
        if isaac_name in aliases:
            candidates.append(str(aliases[isaac_name]))
        if isaac_name in reverse_aliases:
            candidates.append(reverse_aliases[isaac_name])
        actual = next(
            (
                name
                for name in candidates
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
            ),
            None,
        )
        if actual is None:
            raise ValueError(
                f"Isaac body 无法映射到 MuJoCo: expected={isaac_name}, candidates={candidates}"
            )
        resolved.append(actual)
    return resolved


def validate_joint_limits(
    model: mujoco.MjModel, qpos: np.ndarray, addresses: dict[str, int]
) -> None:
    """检查目标 21 关节是否处于 MJCF 物理限位内。"""
    violations = []
    for joint_name, qpos_address in addresses.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if not bool(model.jnt_limited[joint_id]):
            continue
        lower, upper = model.jnt_range[joint_id]
        values = qpos[:, qpos_address]
        count = int(np.count_nonzero((values < lower - 1.0e-6) | (values > upper + 1.0e-6)))
        if count:
            violations.append(f"{joint_name}:{count}")
    if violations:
        raise ValueError(f"CSV 关节违反物理限位: {violations}")


def export_mimic(
    csv_path: Path, metadata_path: Path, config_path: Path, output_path: Path
) -> dict[str, object]:
    """执行完整导出并返回用于终端摘要的 shape 信息。"""
    config_path = config_path.expanduser().resolve()
    config = load_yaml(config_path)
    xml_path = resolve_config_path(config_path, str(config["robot_xml_path"]))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    metadata_path = metadata_path.expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"元数据不存在: path={metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fps = float(metadata.get("fps", 0.0))
    target_fps = float(config["output"]["target_fps"])
    if abs(fps - target_fps) > 1.0e-9:
        raise ValueError(
            "元数据帧率不是已重采样后的目标帧率: "
            f"path={metadata_path}, expected={target_fps}, actual={fps}"
        )
    if int(metadata.get("qpos_size", -1)) != model.nq:
        raise ValueError(
            f"元数据 qpos_size 不匹配: expected={model.nq}, actual={metadata.get('qpos_size')}"
        )
    recorded_xml_sha = metadata.get("robot_xml_sha256")
    current_xml_sha = sha256_file(xml_path)
    if recorded_xml_sha and recorded_xml_sha != current_xml_sha:
        raise ValueError(
            f"XML SHA256 已漂移: path={xml_path}, expected={recorded_xml_sha}, actual={current_xml_sha}"
        )
    recorded_config_sha = metadata.get("config_sha256")
    current_config_sha = sha256_file(config_path)
    if recorded_config_sha and recorded_config_sha != current_config_sha:
        raise ValueError(
            "配置 SHA256 已漂移: "
            f"path={config_path}, expected={recorded_config_sha}, actual={current_config_sha}"
        )

    qpos = load_csv_qpos(csv_path, model)
    if int(metadata.get("num_frames", -1)) != qpos.shape[0]:
        raise ValueError(
            "元数据帧数不匹配: "
            f"expected={qpos.shape[0]}, actual={metadata.get('num_frames')}"
        )
    joint_names = [str(value) for value in config.get("isaac_joint_names", [])]
    body_names = [str(value) for value in config.get("isaac_body_names", [])]
    if joint_names != BUMI3_JOINT_NAMES:
        raise ValueError(
            f"Isaac 关节顺序漂移: expected={BUMI3_JOINT_NAMES}, actual={joint_names}"
        )
    if body_names != BUMI3_ISAAC_BODY_NAMES:
        raise ValueError(
            f"Isaac body 顺序漂移: expected={BUMI3_ISAAC_BODY_NAMES}, actual={body_names}"
        )
    addresses = joint_qpos_addresses(model, joint_names)
    validate_joint_limits(model, qpos, addresses)
    mujoco_body_names = resolve_body_names(
        model, body_names, dict(config.get("mjcf_to_isaac_body_name", {}))
    )
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in mujoco_body_names
    ]

    joint_pos = np.stack([qpos[:, addresses[name]] for name in joint_names], axis=1)
    joint_pos_unwrapped = np.unwrap(joint_pos, axis=0)
    joint_vel = finite_difference(joint_pos_unwrapped, fps)
    body_pos_w = np.empty((qpos.shape[0], len(body_ids), 3), dtype=np.float64)
    body_quat_w = np.empty((qpos.shape[0], len(body_ids), 4), dtype=np.float64)
    for frame_idx, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        body_pos_w[frame_idx] = data.xpos[body_ids]
        body_quat_w[frame_idx] = data.xquat[body_ids]
    body_quat_w = make_quaternions_continuous_wxyz(body_quat_w)
    body_lin_vel_w = finite_difference(body_pos_w, fps)
    body_ang_vel_w = quaternion_angular_velocity_wxyz(body_quat_w, fps)

    anchor_body_name = str(config.get("output", {}).get("anchor_body_name", ""))
    if anchor_body_name not in body_names:
        raise ValueError(
            f"anchor_body_name 必须位于 isaac_body_names: expected one of {body_names}, actual={anchor_body_name}"
        )
    arrays = {
        "fps": np.asarray(fps, dtype=np.float64),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos_w.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float32),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "anchor_body_name": np.asarray(anchor_body_name),
        "source_motion": np.asarray(str(metadata.get("source_motion", csv_path))),
        "robot_name": np.asarray(str(metadata.get("robot", "bumi3"))),
        "quaternion_order": np.asarray("wxyz"),
    }
    for field_name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"导出字段包含 NaN/Inf: field={field_name}, shape={value.shape}")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {key: list(value.shape) for key, value in arrays.items() if value.ndim > 0}


def main() -> None:
    args = parse_args()
    shapes = export_mimic(args.csv, args.metadata, args.config, args.output)
    print(f"BUMI3 Mimic NPZ 已保存: {args.output.expanduser().resolve()}")
    print(json.dumps(shapes, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
