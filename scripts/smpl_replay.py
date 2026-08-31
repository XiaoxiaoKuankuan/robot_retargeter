#!/usr/bin/env python3
"""Replay an AMASS/SMPL motion on the MuJoCo skeleton and export keypoints.

This script loads an AMASS-style motion file, computes SMPL joint poses,
replays them on the project skeleton, and writes retarget-ready keypoints
to output_data/keypoints.

Usage:
	# Run with defaults (open viewer)
	python scripts/smpl_replay.py

	# Generate outputs without opening the viewer
	python scripts/smpl_replay.py --no-viewer

	# Use a specific motion file and playback fps
	python scripts/smpl_replay.py \
		--motion_file dataset/private/test_video_world_params.npz \
		--robot-config config/robot/agibot_x2.yaml \
		--fps 30
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
import threading
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


SMPL_PARENTS = np.array(
	[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19],
	dtype=np.int32,
)

SKELETON_TO_SMPL = {
	"hips": 0,
	"left_up_leg": 1,
	"left_leg": 4,
	"left_foot": 7,
	"left_toe": 10,
	"right_up_leg": 2,
	"right_leg": 5,
	"right_foot": 8,
	"right_toe": 11,
	"spine1": 3,
	"spine2": 6,
	"chest": 9,
	"neck": 12,
	"head": 15,
	"left_shoulder": 13,
	"left_arm": 16,
	"left_fore_arm": 18,
	"left_hand": 20,
	"right_shoulder": 14,
	"right_arm": 17,
	"right_fore_arm": 19,
	"right_hand": 21,
}

DERIVED_BODY_CENTERS = {
	"hips_mean": ("left_up_leg", "right_up_leg"),
	"shoulder_mean": ("left_arm", "right_arm"),
}

REPLAY_BODY_NAMES = list(SKELETON_TO_SMPL.keys()) + list(DERIVED_BODY_CENTERS.keys())

BODY_CHILDREN = {
	"left_up_leg": "left_leg",
	"left_leg": "left_foot",
	"left_foot": "left_toe",
	"right_up_leg": "right_leg",
	"right_leg": "right_foot",
	"right_foot": "right_toe",
	"spine1": "spine2",
	"spine2": "chest",
	"chest": "neck",
	"neck": "head",
	"left_shoulder": "left_arm",
	"left_arm": "left_fore_arm",
	"left_fore_arm": "left_hand",
	"right_shoulder": "right_arm",
	"right_arm": "right_fore_arm",
	"right_fore_arm": "right_hand",
}

BODY_LOCAL_DIRECTIONS = {
	"left_up_leg": np.array([0.0, -1.0, 0.0], dtype=np.float32),
	"left_leg": np.array([0.0, -1.0, 0.0], dtype=np.float32),
	"left_foot": np.array([0.0, -0.054, 0.125], dtype=np.float32),
	"right_up_leg": np.array([0.0, -1.0, 0.0], dtype=np.float32),
	"right_leg": np.array([0.0, -1.0, 0.0], dtype=np.float32),
	"right_foot": np.array([0.0, -0.054, 0.125], dtype=np.float32),
	"spine1": np.array([0.0, 1.0, 0.0], dtype=np.float32),
	"spine2": np.array([0.0, 1.0, 0.0], dtype=np.float32),
	"chest": np.array([0.0, 1.0, 0.0], dtype=np.float32),
	"neck": np.array([0.0, 1.0, 0.0], dtype=np.float32),
	"left_shoulder": np.array([1.0, 0.0, 0.0], dtype=np.float32),
	"left_arm": np.array([1.0, 0.0, 0.0], dtype=np.float32),
	"left_fore_arm": np.array([1.0, 0.0, 0.0], dtype=np.float32),
	"right_shoulder": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
	"right_arm": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
	"right_fore_arm": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
}

DEFAULT_ROBOT_CONFIG_PATH = Path("config/robot/t800.yaml")
DEFAULT_SKELETON_CONFIG_PATH = Path("config/skeleton/skeleton.yaml")
RETARGET_KEYPOINT_RGBA = np.array([0.1, 0.9, 1.0, 0.85], dtype=np.float32)
RETARGET_KEYPOINT_RADIUS = 0.05
RETARGET_BLEND_FOOT_RGBA = np.array([0.2, 1.0, 0.2, 0.95], dtype=np.float32)
RETARGET_BLEND_FOOT_RADIUS = 0.06
RETARGET_BLEND_KNEE_RGBA = np.array([0.2, 1.0, 0.2, 0.95], dtype=np.float32)
RETARGET_BLEND_KNEE_RADIUS = 0.055
CONTACT_ACTIVE_RGBA = np.array([0.1, 0.35, 1.0, 0.95], dtype=np.float32)
CONTACT_ACTIVE_RADIUS = 0.06
RETARGET_KEYPOINT_AXIS_RADIUS = 0.01
RETARGET_KEYPOINT_AXIS_HALF_LENGTH = 0.08
RETARGET_KEYPOINT_AXIS_COLORS = np.array(
	[
		[1.0, 0.0, 0.0, 1.0],
		[0.0, 1.0, 0.0, 1.0],
		[0.0, 0.0, 1.0, 1.0],
	],
	dtype=np.float32,
)
ROT_X_NEG_90 = Rotation.from_euler("x", -90.0, degrees=True).as_matrix().astype(np.float32)
ROT_Y_POS_90 = Rotation.from_euler("y", 90.0, degrees=True).as_matrix().astype(np.float32)


def iter_progress(
	iterable,
	*,
	total: int | None = None,
	desc: str | None = None,
	unit: str | None = None,
):
	return tqdm(
		iterable,
		total=total,
		desc=desc,
		unit=unit,
		dynamic_ncols=True,
		leave=True,
	)


def batch_rotation_between_vectors_wxyz(
	v_from: np.ndarray,
	v_to: np.ndarray,
	eps: float = 1e-8,
) -> np.ndarray:
	"""Compute the minimal world-frame rotation (wxyz quaternions) that maps each
	``v_from`` direction onto the corresponding ``v_to`` direction.

	The rotation axis is the cross product of the two vectors, i.e. the normal of
	the plane spanned by them, so this is exactly a rotation about that plane normal.
	"""
	v_from = np.asarray(v_from, dtype=np.float64)
	v_to = np.asarray(v_to, dtype=np.float64)
	from_norm = np.linalg.norm(v_from, axis=-1, keepdims=True)
	to_norm = np.linalg.norm(v_to, axis=-1, keepdims=True)
	from_u = v_from / np.clip(from_norm, eps, None)
	to_u = v_to / np.clip(to_norm, eps, None)

	axis = np.cross(from_u, to_u)
	axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
	cos_angle = np.clip(np.sum(from_u * to_u, axis=-1, keepdims=True), -1.0, 1.0)
	angle = np.arccos(cos_angle)

	# Pick an arbitrary perpendicular axis for the (near-)antiparallel case so the
	# 180-degree flip stays well defined.
	fallback_axis = np.cross(from_u, np.array([1.0, 0.0, 0.0], dtype=np.float64))
	fallback_axis_norm = np.linalg.norm(fallback_axis, axis=-1, keepdims=True)
	fallback_axis = np.where(
		fallback_axis_norm > eps,
		fallback_axis,
		np.cross(from_u, np.array([0.0, 1.0, 0.0], dtype=np.float64)),
	)
	axis = np.where(axis_norm > eps, axis, fallback_axis)
	axis = axis / np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), eps, None)

	rotvec = axis * angle
	quat_xyzw = Rotation.from_rotvec(rotvec).as_quat()
	return quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)


def multiply_quaternions_wxyz(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
	"""Compose two batches of wxyz quaternions as ``q_left * q_right``."""
	left_xyzw = np.asarray(q_left, dtype=np.float64)[:, [1, 2, 3, 0]]
	right_xyzw = np.asarray(q_right, dtype=np.float64)[:, [1, 2, 3, 0]]
	result_xyzw = (Rotation.from_quat(left_xyzw) * Rotation.from_quat(right_xyzw)).as_quat()
	return result_xyzw[:, [3, 0, 1, 2]].astype(np.float32)


def compute_two_bone_ik_knee_positions(
	hip_positions: np.ndarray,
	knee_positions: np.ndarray,
	foot_positions: np.ndarray,
	target_foot_positions: np.ndarray,
) -> np.ndarray:
	upper_leg = knee_positions - hip_positions
	lower_leg = foot_positions - knee_positions
	upper_len = np.linalg.norm(upper_leg, axis=-1)
	lower_len = np.linalg.norm(lower_leg, axis=-1)

	original_hip_to_foot = foot_positions - hip_positions
	original_hip_to_foot_len = np.linalg.norm(original_hip_to_foot, axis=-1)
	original_u = original_hip_to_foot / np.clip(original_hip_to_foot_len[:, None], 1e-8, None)

	knee_from_hip = knee_positions - hip_positions
	knee_along_u = np.sum(knee_from_hip * original_u, axis=-1, keepdims=True)
	knee_projection = hip_positions + knee_along_u * original_u
	bend_pref = knee_positions - knee_projection
	bend_pref_norm = np.linalg.norm(bend_pref, axis=-1, keepdims=True)
	world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
	bend_pref = np.where(
		bend_pref_norm > 1e-8,
		bend_pref,
		np.cross(original_u, world_up[None, :]),
	)
	bend_pref_norm = np.linalg.norm(bend_pref, axis=-1, keepdims=True)
	bend_pref = np.where(
		bend_pref_norm > 1e-8,
		bend_pref,
		np.cross(original_u, np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, :]),
	)
	bend_pref = bend_pref / np.clip(np.linalg.norm(bend_pref, axis=-1, keepdims=True), 1e-8, None)

	# Keep blended knee in the original hip-knee-foot plane whenever possible.
	plane_normal = np.cross(upper_leg, original_hip_to_foot)
	plane_normal_norm = np.linalg.norm(plane_normal, axis=-1, keepdims=True)
	plane_normal = np.where(
		plane_normal_norm > 1e-8,
		plane_normal,
		np.cross(original_u, bend_pref),
	)
	plane_normal_norm = np.linalg.norm(plane_normal, axis=-1, keepdims=True)
	plane_normal = np.where(
		plane_normal_norm > 1e-8,
		plane_normal,
		np.cross(original_u, world_up[None, :]),
	)
	plane_normal_norm = np.linalg.norm(plane_normal, axis=-1, keepdims=True)
	plane_normal = np.where(
		plane_normal_norm > 1e-8,
		plane_normal,
		np.cross(original_u, np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, :]),
	)
	plane_normal = plane_normal / np.clip(np.linalg.norm(plane_normal, axis=-1, keepdims=True), 1e-8, None)

	target_hip_to_foot = target_foot_positions - hip_positions
	target_hip_to_foot_len = np.linalg.norm(target_hip_to_foot, axis=-1)
	target_u = target_hip_to_foot / np.clip(target_hip_to_foot_len[:, None], 1e-8, None)

	d = np.clip(target_hip_to_foot_len, 1e-8, upper_len + lower_len - 1e-8)
	x = (upper_len**2 - lower_len**2 + d**2) / (2.0 * np.clip(d, 1e-8, None))
	h = np.sqrt(np.clip(upper_len**2 - x**2, 0.0, None))

	bend_pref_proj = bend_pref - np.sum(bend_pref * target_u, axis=-1, keepdims=True) * target_u
	bend_dir = np.cross(plane_normal, target_u)
	bend_dir_norm = np.linalg.norm(bend_dir, axis=-1, keepdims=True)
	bend_dir = np.where(
		bend_dir_norm > 1e-8,
		bend_dir,
		bend_pref_proj,
	)
	bend_dir_norm = np.linalg.norm(bend_dir, axis=-1, keepdims=True)
	bend_dir = np.where(
		bend_dir_norm > 1e-8,
		bend_dir,
		np.cross(target_u, np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, :]),
	)
	bend_dir = bend_dir / np.clip(np.linalg.norm(bend_dir, axis=-1, keepdims=True), 1e-8, None)

	bend_pref_proj_norm = np.linalg.norm(bend_pref_proj, axis=-1, keepdims=True)
	bend_pref_proj = np.where(
		bend_pref_proj_norm > 1e-8,
		bend_pref_proj,
		bend_pref,
	)
	bend_sign = np.where(np.sum(bend_dir * bend_pref_proj, axis=-1) >= 0.0, 1.0, -1.0)
	bend_dir = bend_dir * bend_sign[:, None]

	blended_knee_positions = hip_positions + x[:, None] * target_u + h[:, None] * bend_dir
	return blended_knee_positions.astype(np.float32)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Replay an AMASS/SMPL motion on the Retarget MuJoCo skeleton."
	)
	parser.add_argument(
		"--motion_file",
		type=Path,
		default=Path("dataset/ACCAD/Form_1_stageii.npz"),
		help="Path to an AMASS-style .npz file with trans/root_orient/pose_body.",
	)
	parser.add_argument(
		"--mjcf",
		type=Path,
		default=Path("asset/skeleton/mjcf/skeleton.xml"),
		help="Path to the MuJoCo skeleton MJCF file.",
	)
	parser.add_argument(
		"--smpl-model-path",
		type=Path,
		default=Path("asset/smplx/SMPLX_NEUTRAL.npz"),
		help="Directory containing SMPL_{GENDER}.pkl files, or one of those files.",
	)
	parser.add_argument(
		"--gender",
		choices=["female", "male", "neutral", "auto"],
		default="auto", 
		help="Override the SMPL gender. Default uses the motion file metadata.",
	)
	parser.add_argument(
		"--device",
		default="cpu",
		help="Torch device used for SMPL forward passes.",
	)
	parser.add_argument(
		"--start-frame",
		type=int,
		default=0,
		help="First frame to replay.",
	)
	parser.add_argument(
		"--end-frame",
		type=int,
		default=-1,
		help="Last frame to replay, inclusive. Default replays until the end.",
	)
	parser.add_argument(
		"--stride",
		type=int,
		default=1,
		help="Replay every Nth frame.",
	)
	parser.add_argument(
		"--fps",
		type=float,
		default=0.0,
		help="Override playback fps. Default uses mocap_frame_rate from the motion file.",
	)
	parser.add_argument(
		"--model-type",
		choices=["auto", "smpl", "smplx"],
		default="auto",
		help="输入人体模型类型；auto 优先读取文件元数据，缺失时使用 smpl。",
	)
	parser.add_argument(
		"--translation-scale",
		type=float,
		default=1.0,
		help="显式平移缩放；默认 1.0，不自动猜测米/毫米。",
	)
	parser.add_argument(
		"--up-axis",
		choices=["auto", "y", "z"],
		default="auto",
		help="输入世界坐标的向上轴；y 会显式转为 MuJoCo 的 z-up。",
	)
	parser.add_argument(
		"--target-fps",
		type=float,
		default=0.0,
		help="真实重采样目标频率；0 保持源频率。",
	)
	parser.add_argument(
		"--max-frames",
		type=int,
		default=0,
		help="重采样后最多保留的帧数；0 表示完整动作。",
	)
	parser.add_argument(
		"--keypoints-name",
		default=None,
		help="覆盖输出 keypoint 文件 stem；默认使用输入动作文件 stem。",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("output_data"),
		help="输出根目录；keypoint 保存到 <目录>/keypoints/<机器人>/。",
	)
	parser.add_argument(
		"--loop",
		type=bool,
		default=True,
		help="Loop the clip until the viewer window closes.",
	)
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=256,
		help="Number of frames processed per SMPL forward chunk.",
	)
	parser.add_argument(
		"--translation-offset",
		type=float,
		nargs=3,
		default=(0.0, 0.0, 0.0),
		metavar=("X", "Y", "Z"),
		help="Offset added to all replayed joint positions.",
	)
	parser.add_argument(
		"--print-summary",
		action="store_true",
		help="Print clip metadata before launching the viewer.",
	)
	parser.add_argument(
		"--no-viewer",
		action="store_true",
		help="Generate outputs without launching the MuJoCo viewer.",
	)
	parser.add_argument(
		"--robot-config",
		type=Path,
		default=DEFAULT_ROBOT_CONFIG_PATH,
		help="Path to the robot link config YAML used for zero-pose link lengths.",
	)
	parser.add_argument(
		"--skeleton-config",
		type=Path,
		default=DEFAULT_SKELETON_CONFIG_PATH,
		help="Path to the skeleton link config YAML used to compute scale factors.",
	)
	return parser.parse_args()


def load_link_pairs_config(config_path: Path, section_name: str) -> dict[str, tuple[str, str]]:
	if not config_path.exists():
		raise FileNotFoundError(f"Link config file not found: {config_path}")
	with config_path.open("r", encoding="utf-8") as file:
		config = yaml.safe_load(file) or {}
	section = config.get(section_name)
	if not isinstance(section, dict) or not section:
		raise ValueError(f"配置段必须为非空映射: path={config_path}, field={section_name}")
	link_pairs: dict[str, tuple[str, str]] = {}
	for link_name, body_spec in section.items():
		if not isinstance(body_spec, (list, tuple)) or len(body_spec) != 2:
			raise ValueError(
				f"连杆端点必须恰好两个 body: path={config_path}, "
				f"field={section_name}.{link_name}, actual={body_spec}"
			)
		body_names = tuple(str(name).strip() for name in body_spec)
		if not all(body_names):
			raise ValueError(
				f"连杆端点名称不得为空: path={config_path}, field={section_name}.{link_name}"
			)
		link_pairs[str(link_name)] = (body_names[0], body_names[1])
	return link_pairs


def load_robot_links_config(config_path: Path) -> dict[str, tuple[str, str]]:
	return load_link_pairs_config(config_path, "robot_links")


def load_skeleton_links_config(config_path: Path) -> dict[str, tuple[str, str]]:
	return load_link_pairs_config(config_path, "skeleton_links")


def derive_leg_body_chain_from_links(
	skeleton_links: dict[str, tuple[str, str]],
	leg_side: str,
) -> tuple[str, str, str]:
	thigh_link_name = f"{leg_side}_thigh"
	calf_link_name = f"{leg_side}_calf"
	if thigh_link_name not in skeleton_links:
		raise ValueError(f"Missing skeleton link required for leg chain: {thigh_link_name}")
	if calf_link_name not in skeleton_links:
		raise ValueError(f"Missing skeleton link required for leg chain: {calf_link_name}")

	thigh_parent, thigh_child = skeleton_links[thigh_link_name]
	calf_parent, calf_child = skeleton_links[calf_link_name]
	if thigh_child != calf_parent:
		raise ValueError(
			f"Inconsistent {leg_side} leg links: {thigh_link_name} ends at {thigh_child}, "
			f"but {calf_link_name} starts at {calf_parent}"
		)
	return thigh_parent, thigh_child, calf_child


def load_scalar_float_config(config_path: Path, field_name: str, default: float | None = None) -> float:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	for raw_line in config_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].strip()
		if not line:
			continue
		key, separator, value = line.partition(":")
		if separator and key.strip() == field_name:
			parsed = value.strip()
			if not parsed:
				raise ValueError(f"Missing value for '{field_name}' in: {config_path}")
			return float(parsed)

	if default is not None:
		return float(default)
	raise ValueError(f"Missing required field '{field_name}' in: {config_path}")


def load_body_chain_config(config_path: Path, field_name: str, expected_len: int) -> tuple[str, ...]:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	for raw_line in config_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].strip()
		if not line:
			continue
		key, separator, value = line.partition(":")
		if not separator or key.strip() != field_name:
			continue

		value = value.strip()
		if not (value.startswith("[") and value.endswith("]")):
			raise ValueError(f"Invalid list for '{field_name}' in: {config_path}")
		items = tuple(item.strip() for item in value[1:-1].split(",") if item.strip())
		if len(items) != expected_len:
			raise ValueError(
				f"'{field_name}' must contain exactly {expected_len} body names in: {config_path}"
			)
		return items

	raise ValueError(f"Missing required field '{field_name}' in: {config_path}")


def load_body_list_config(config_path: Path, field_name: str) -> tuple[str, ...]:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	for raw_line in config_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].strip()
		if not line:
			continue
		key, separator, value = line.partition(":")
		if not separator or key.strip() != field_name:
			continue

		value = value.strip()
		if not (value.startswith("[") and value.endswith("]")):
			raise ValueError(f"Invalid list for '{field_name}' in: {config_path}")
		items = tuple(item.strip() for item in value[1:-1].split(",") if item.strip())
		if not items:
			raise ValueError(f"'{field_name}' must contain at least one body name in: {config_path}")
		return items

	raise ValueError(f"Missing required field '{field_name}' in: {config_path}")


def load_yaml_body_list_config(config_path: Path, field_name: str) -> tuple[str, ...]:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with config_path.open("r", encoding="utf-8") as f:
		config = yaml.safe_load(f) or {}
	value = config.get(field_name)
	if not isinstance(value, list) or not value:
		raise ValueError(f"Missing or invalid list field '{field_name}' in: {config_path}")
	items = tuple(str(item).strip() for item in value if str(item).strip())
	if not items:
		raise ValueError(f"'{field_name}' must contain at least one body name in: {config_path}")
	return items


def canonicalize_contact_name(body_name: str) -> str:
	lower = body_name.lower()
	if "left" in lower and "toe" in lower:
		return "left_toe"
	if "right" in lower and "toe" in lower:
		return "right_toe"
	if "left" in lower and "foot" in lower and "end" in lower:
		return "left_foot_end"
	if "right" in lower and "foot" in lower and "end" in lower:
		return "right_foot_end"
	if "left" in lower and ("hand" in lower or "wrist" in lower):
		return "left_hand"
	if "right" in lower and ("hand" in lower or "wrist" in lower):
		return "right_hand"
	return body_name


def load_scalar_int_config(config_path: Path, field_name: str, default: int | None = None) -> int:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	for raw_line in config_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].strip()
		if not line:
			continue
		key, separator, value = line.partition(":")
		if separator and key.strip() == field_name:
			parsed = value.strip()
			if not parsed:
				raise ValueError(f"Missing value for '{field_name}' in: {config_path}")
			return int(parsed)

	if default is not None:
		return int(default)
	raise ValueError(f"Missing required field '{field_name}' in: {config_path}")


def load_path_config(config_path: Path, field_name: str, default: Path | None = None) -> Path:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	for raw_line in config_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].strip()
		if not line:
			continue
		key, separator, value = line.partition(":")
		if separator and key.strip() == field_name:
			parsed = value.strip()
			if not parsed:
				raise ValueError(f"Missing value for '{field_name}' in: {config_path}")
			if parsed[:1] in {'"', "'"} and parsed[-1:] == parsed[:1]:
				parsed = parsed[1:-1]
			return Path(parsed)

	if default is not None:
		return default
	raise ValueError(f"Missing required field '{field_name}' in: {config_path}")


def load_euler_offset_map_config(config_path: Path, section_name: str) -> dict[str, np.ndarray]:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	offset_map: dict[str, np.ndarray] = {}
	in_section = False
	for raw_line in config_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.split("#", 1)[0].rstrip()
		if not line.strip():
			continue

		if not in_section:
			if line.strip() == f"{section_name}:":
				in_section = True
			continue

		if not raw_line[:1].isspace():
			break

		stripped = line.strip()
		name, separator, values_str = stripped.partition(":")
		if not separator:
			raise ValueError(f"Invalid {section_name} entry: {raw_line}")
		values_str = values_str.strip()
		if not (values_str.startswith("[") and values_str.endswith("]")):
			raise ValueError(f"Invalid {section_name} euler list: {raw_line}")
		parts = [item.strip() for item in values_str[1:-1].split(",") if item.strip()]
		if len(parts) != 3:
			raise ValueError(f"{section_name} entry must contain exactly 3 values: {raw_line}")
		offset_map[name.strip()] = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)

	if not offset_map:
		raise ValueError(f"No {section_name} entries found in: {config_path}")
	return offset_map


def _parse_float_list(value: str, expected_len: int, context: str) -> np.ndarray:
	value = value.strip()
	if not (value.startswith("[") and value.endswith("]")):
		raise ValueError(f"Invalid list for {context}: {value}")
	parts = [item.strip() for item in value[1:-1].split(",") if item.strip()]
	if len(parts) != expected_len:
		raise ValueError(f"{context} must contain exactly {expected_len} values: {value}")
	return np.asarray([float(item) for item in parts], dtype=np.float32)


def load_key_frame_config(
	config_path: Path,
	section_name: str = "key_frame_config",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with config_path.open("r", encoding="utf-8") as file:
		config = yaml.safe_load(file) or {}
	section = config.get(section_name)
	if not isinstance(section, dict) or not section:
		raise ValueError(f"配置段必须为非空映射: path={config_path}, field={section_name}")
	offset_map: dict[str, np.ndarray] = {}
	axis_map: dict[str, np.ndarray] = {}
	for body_name, entry in section.items():
		if not isinstance(entry, dict):
			raise ValueError(
				f"关键帧配置必须为映射: path={config_path}, field={section_name}.{body_name}, actual={entry}"
			)
		offset = np.asarray(entry.get("offset_deg_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)
		if offset.shape != (3,) or not np.all(np.isfinite(offset)):
			raise ValueError(
				f"关键帧欧拉偏置必须为 3 个有限值: path={config_path}, "
				f"field={section_name}.{body_name}.offset_deg_xyz, actual={offset.shape}"
			)
		axes = entry.get("axis_map_cols")
		if not isinstance(axes, dict) or set(axes) != {"x", "y", "z"}:
			raise ValueError(
				f"关键帧轴映射必须含 x/y/z: path={config_path}, "
				f"field={section_name}.{body_name}.axis_map_cols, actual={axes}"
			)
		columns = [np.asarray(axes[axis], dtype=np.float32) for axis in ("x", "y", "z")]
		if any(column.shape != (3,) or not np.all(np.isfinite(column)) for column in columns):
			raise ValueError(
				f"关键帧轴列必须为 3 个有限值: path={config_path}, "
				f"field={section_name}.{body_name}.axis_map_cols"
			)
		matrix = np.column_stack(columns).astype(np.float32)
		if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-5) or not np.isclose(
			np.linalg.det(matrix), 1.0, atol=1.0e-5
		):
			raise ValueError(
				f"关键帧轴映射必须为右手正交矩阵: path={config_path}, "
				f"field={section_name}.{body_name}.axis_map_cols, det={np.linalg.det(matrix)}"
			)
		offset_map[str(body_name)] = offset
		axis_map[str(body_name)] = matrix
	return offset_map, axis_map


def compute_robot_link_lengths(
	config_path: Path,
	robot_mjcf_path: Path,
) -> dict[str, float]:
	robot_links = load_robot_links_config(config_path)
	if not robot_mjcf_path.exists():
		raise FileNotFoundError(f"Robot MJCF file not found: {robot_mjcf_path}")

	model = mujoco.MjModel.from_xml_path(str(robot_mjcf_path))
	data = mujoco.MjData(model)
	mujoco.mj_forward(model, data)

	link_lengths: dict[str, float] = {}
	for link_name, (start_body, end_body) in robot_links.items():
		start_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, start_body)
		if start_body_id < 0:
			raise ValueError(f"Missing robot body in MJCF: {start_body}")
		end_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_body)
		if end_body_id < 0:
			raise ValueError(f"Missing robot body in MJCF: {end_body}")

		start_pos = data.xpos[start_body_id]
		end_pos = data.xpos[end_body_id]
		link_lengths[link_name] = float(np.linalg.norm(end_pos - start_pos))

	return link_lengths


def compute_robot_body_local_offset(
	robot_mjcf_path: Path,
	anchor_body: str,
	target_body: str,
) -> np.ndarray:
	if not robot_mjcf_path.exists():
		raise FileNotFoundError(f"Robot MJCF file not found: {robot_mjcf_path}")

	model = mujoco.MjModel.from_xml_path(str(robot_mjcf_path))
	data = mujoco.MjData(model)
	mujoco.mj_forward(model, data)

	anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body)
	if anchor_body_id < 0:
		raise ValueError(f"Missing robot anchor body in MJCF: {anchor_body}")
	target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
	if target_body_id < 0:
		raise ValueError(f"Missing robot target body in MJCF: {target_body}")

	anchor_pos = data.xpos[anchor_body_id]
	target_pos = data.xpos[target_body_id]
	anchor_rot = data.xmat[anchor_body_id].reshape(3, 3)
	local_offset = anchor_rot.T @ (target_pos - anchor_pos)
	return local_offset.astype(np.float32)


def quat_rotate_vectors_wxyz(quaternions: np.ndarray, vector: np.ndarray) -> np.ndarray:
	vectors = np.broadcast_to(vector.astype(np.float32), (quaternions.shape[0], 3))
	rotated = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).apply(vectors)
	return rotated.astype(np.float32)


def load_mjcf_nested_body_offsets(
	mjcf_path: Path,
	body_names: tuple[str, ...],
) -> dict[str, tuple[str, np.ndarray]]:
	if not body_names:
		return {}
	if not mjcf_path.exists():
		raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")

	model = mujoco.MjModel.from_xml_path(str(mjcf_path))
	data = mujoco.MjData(model)
	mujoco.mj_forward(model, data)

	offsets: dict[str, tuple[str, np.ndarray]] = {}
	for body_name in body_names:
		body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
		if body_id < 0:
			raise ValueError(f"Missing contact body in MJCF: {body_name}")
		parent_id = int(model.body_parentid[body_id])
		if parent_id <= 0:
			raise ValueError(f"Contact body must have a parent body in MJCF: {body_name}")
		parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
		if parent_name is None:
			raise ValueError(f"Missing parent body name in MJCF for: {body_name}")
		parent_pos = data.xpos[parent_id]
		body_pos = data.xpos[body_id]
		parent_rot = data.xmat[parent_id].reshape(3, 3)
		local_offset = parent_rot.T @ (body_pos - parent_pos)
		offsets[body_name] = (parent_name, local_offset.astype(np.float32))

	return offsets


def resolve_contact_positions(
	positions: np.ndarray,
	quaternions: np.ndarray,
	contact_links: tuple[str, ...],
	skeleton_mjcf_path: Path,
) -> np.ndarray:
	body_slots = {name: idx for idx, name in enumerate(REPLAY_BODY_NAMES)}
	nested_body_names = tuple(body_name for body_name in contact_links if body_name not in body_slots)
	nested_body_offsets = load_mjcf_nested_body_offsets(skeleton_mjcf_path, nested_body_names)
	contact_positions = np.zeros((positions.shape[0], len(contact_links), 3), dtype=np.float32)

	for contact_idx, body_name in enumerate(contact_links):
		if body_name in body_slots:
			contact_positions[:, contact_idx, :] = positions[:, body_slots[body_name], :]
			continue

		if body_name not in nested_body_offsets:
			raise ValueError(f"Missing contact body in replay buffers and MJCF: {body_name}")
		parent_name, local_offset = nested_body_offsets[body_name]
		if parent_name not in body_slots:
			raise ValueError(f"Missing contact parent body in replay buffers: {parent_name}")
		parent_idx = body_slots[parent_name]
		contact_positions[:, contact_idx, :] = (
			positions[:, parent_idx, :] + quat_rotate_vectors_wxyz(quaternions[:, parent_idx, :], local_offset)
		)

	return contact_positions


def compute_windowed_point_speeds(
	point_positions: np.ndarray,
	fps: float,
	window: int,
	progress_desc: str | None = None,
) -> np.ndarray:
	if fps <= 0.0:
		raise ValueError(f"FPS must be positive, got {fps}")
	if window <= 0:
		raise ValueError(f"Window must be positive, got {window}")

	num_frames = point_positions.shape[0]
	half_window = max(1, window // 2)
	speeds = np.zeros(point_positions.shape[:2], dtype=np.float32)

	frame_iter = range(num_frames)
	if progress_desc is not None:
		frame_iter = iter_progress(frame_iter, total=num_frames, desc=progress_desc, unit="frame")
	for frame_idx in frame_iter:
		start_idx = max(0, frame_idx - half_window)
		end_idx = min(num_frames - 1, frame_idx + half_window)
		frame_delta = end_idx - start_idx
		if frame_delta <= 0:
			continue
		displacement = point_positions[end_idx, :, :] - point_positions[start_idx, :, :]
		speeds[frame_idx, :] = np.linalg.norm(displacement, axis=-1) / (frame_delta / fps)

	return speeds


def compute_contact_sequence(
	positions: np.ndarray,
	quaternions: np.ndarray,
	contact_links: tuple[str, ...],
	skeleton_mjcf_path: Path,
	fps: float,
	vel_window: int,
	vel_threshold: float,
	height_threshold: float,
	height_reference_contact_names: tuple[str, ...] | None = None,
	height_floor_percentile: float | None = None,
	height_floor_estimator: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	contact_positions = resolve_contact_positions(
		positions=positions,
		quaternions=quaternions,
		contact_links=contact_links,
		skeleton_mjcf_path=skeleton_mjcf_path,
	)
	contact_speeds = compute_windowed_point_speeds(
		contact_positions,
		fps=fps,
		window=vel_window,
		progress_desc="Contact speed",
	)
	height_values = contact_positions[:, :, 2]
	if height_floor_percentile is not None or height_floor_estimator is not None:
		reference_names = height_reference_contact_names or contact_links
		contact_index = {name: index for index, name in enumerate(contact_links)}
		missing = [name for name in reference_names if name not in contact_index]
		if missing:
			raise ValueError(
				f"接触地板 reference 不在 contact_links: missing={missing}, actual={contact_links}"
			)
		reference_indices = [contact_index[name] for name in reference_names]
		if height_floor_estimator is not None:
			floor_height, _ = fit_stable_support_floor_height(
				contact_positions=contact_positions,
				contact_speeds=contact_speeds,
				reference_indices=np.asarray(reference_indices, dtype=np.int32),
				speed_threshold_mps=float(
					height_floor_estimator["speed_threshold_mps"]
				),
				inlier_tolerance_m=float(
					height_floor_estimator["inlier_tolerance_m"]
				),
				minimum_samples=int(height_floor_estimator["minimum_samples"]),
			)
		else:
			floor_height = compute_sequence_contact_floor_height(
				contact_positions=contact_positions,
				reference_indices=np.asarray(reference_indices, dtype=np.int32),
				percentile=float(height_floor_percentile),
			)
		height_values = height_values - floor_height
	contact_states = np.logical_and(
		contact_speeds <= float(vel_threshold),
		height_values <= float(height_threshold),
	)
	return contact_positions, contact_speeds, contact_states.astype(np.bool_)


def stabilize_contact_states(
	contact_speeds: np.ndarray,
	relative_heights: np.ndarray,
	enter_velocity: float,
	exit_velocity: float,
	enter_height: float,
	exit_height: float,
	enter_confirmation_frames: int,
	exit_confirmation_frames: int,
	minimum_contact_frames: int,
	minimum_swing_frames: int,
) -> np.ndarray:
	"""用施密特阈值和最短持续时间稳定逐点接触状态。

	进入接触使用更严格的速度/高度阈值，保持接触使用更宽松的退出阈值，避免脚点
	在阈值附近逐帧抖动。进入和退出都需要连续确认；状态切换后还必须满足最短接触
	或摆动帧数。进入确认完成后会把连续满足条件的确认窗口回填为接触，使离线轨迹
	不会平白丢失真正落地的头几帧；退出确认窗口仍保留接触，直到确认离地为止。
	"""
	contact_speeds = np.asarray(contact_speeds, dtype=np.float64)
	relative_heights = np.asarray(relative_heights, dtype=np.float64)
	if contact_speeds.shape != relative_heights.shape or contact_speeds.ndim != 2:
		raise ValueError(
			"接触速度与相对高度必须是相同的 [T,C]: "
			f"speed={contact_speeds.shape}, height={relative_heights.shape}"
		)
	thresholds = np.asarray(
		[enter_velocity, exit_velocity, enter_height, exit_height], dtype=np.float64
	)
	if not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0.0):
		raise ValueError(f"接触滞回阈值必须是非负有限值: {thresholds.tolist()}")
	if exit_velocity < enter_velocity or exit_height < enter_height:
		raise ValueError(
			"退出阈值不得严于进入阈值: "
			f"velocity={enter_velocity}/{exit_velocity}, height={enter_height}/{exit_height}"
		)
	frame_parameters = (
		enter_confirmation_frames,
		exit_confirmation_frames,
		minimum_contact_frames,
		minimum_swing_frames,
	)
	if any(int(value) < 1 for value in frame_parameters):
		raise ValueError(f"接触滞回帧参数必须全部 >=1: {frame_parameters}")

	enter_candidate = (contact_speeds <= enter_velocity) & (
		relative_heights <= enter_height
	)
	stay_candidate = (contact_speeds <= exit_velocity) & (
		relative_heights <= exit_height
	)
	states = np.zeros(contact_speeds.shape, dtype=np.bool_)
	for contact_idx in range(contact_speeds.shape[1]):
		active = False
		active_duration = 0
		inactive_duration = int(minimum_swing_frames)
		enter_count = 0
		exit_count = 0
		for frame_idx in range(contact_speeds.shape[0]):
			if active:
				states[frame_idx, contact_idx] = True
				active_duration += 1
				if stay_candidate[frame_idx, contact_idx]:
					exit_count = 0
				elif active_duration >= int(minimum_contact_frames):
					exit_count += 1
				if exit_count >= int(exit_confirmation_frames):
					states[frame_idx, contact_idx] = False
					active = False
					inactive_duration = 1
					active_duration = 0
					enter_count = 0
					exit_count = 0
				continue

			inactive_duration += 1
			if (
				inactive_duration >= int(minimum_swing_frames)
				and enter_candidate[frame_idx, contact_idx]
			):
				enter_count += 1
			else:
				enter_count = 0
			if enter_count >= int(enter_confirmation_frames):
				start = frame_idx - enter_count + 1
				states[start : frame_idx + 1, contact_idx] = True
				active = True
				active_duration = enter_count
				inactive_duration = 0
				exit_count = 0
	return states


def compute_sequence_contact_floor_height(
	contact_positions: np.ndarray,
	reference_indices: np.ndarray,
	percentile: float,
) -> float:
	"""用指定接触点的全序列稳健分位估计人体动作地板高度。

	该值同时服务于接触检测和关键点初始地面偏移。这样即使动作开头处于腾空或
	快速移动、尚未出现激活接触帧，也不会让地面偏移错误地从世界零高度开始。
	"""
	if contact_positions.ndim != 3 or contact_positions.shape[-1] != 3:
		raise ValueError(
			"contact_positions shape must be [T, C, 3], got "
			f"{contact_positions.shape}"
		)
	indices = np.asarray(reference_indices, dtype=np.int32)
	if indices.ndim != 1 or indices.size == 0:
		raise ValueError(f"reference_indices 必须是一维非空数组: actual={indices.shape}")
	if np.any(indices < 0) or np.any(indices >= contact_positions.shape[1]):
		raise ValueError(
			f"reference_indices 越界: contact_count={contact_positions.shape[1]}, "
			f"actual={indices.tolist()}"
		)
	percentile = float(percentile)
	if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
		raise ValueError(f"接触地板分位必须位于 [0,100]: actual={percentile}")
	return float(np.percentile(contact_positions[:, indices, 2], percentile))


def fit_stable_support_floor_height(
	contact_positions: np.ndarray,
	contact_speeds: np.ndarray,
	reference_indices: np.ndarray,
	speed_threshold_mps: float,
	inlier_tolerance_m: float,
	minimum_samples: int,
) -> tuple[float, dict[str, float | int | str]]:
	"""从低速足点的最低密集高度簇稳健拟合一条恒定水平地板。

	该估计器先只保留速度不超过 ``speed_threshold_mps`` 的足点样本，再在排序后的
	高度上用双指针寻找宽度不超过 ``2 * inlier_tolerance_m`` 的最大密集区间。若
	多个区间样本数相同则选择高度更低者，避免把长时间静止但抬起的脚误认为地板。
	最后以候选中心 ``inlier_tolerance_m`` 内样本的中位数作为地板高度。中位数对
	簇内少量穿地或漂浮异常具有 50% breakdown point；整个流程不再依赖全序列最低
	1% 样本，也不会让单个极低异常帧决定整段高度。

	当前播放器地面是水平 ``z=0``，因此这里只拟合常数而不是斜平面。对斜坡数据
	应显式扩展为带法向约束的稳健平面拟合，不能把斜率悄悄混入根高度偏移。
	"""
	if contact_positions.ndim != 3 or contact_positions.shape[-1] != 3:
		raise ValueError(
			"contact_positions shape must be [T,C,3], got "
			f"{contact_positions.shape}"
		)
	if contact_speeds.shape != contact_positions.shape[:2]:
		raise ValueError(
			"contact_speeds shape must match contact_positions[:2], got "
			f"{contact_speeds.shape} and {contact_positions.shape}"
		)
	indices = np.asarray(reference_indices, dtype=np.int32)
	if indices.ndim != 1 or indices.size == 0:
		raise ValueError(f"reference_indices 必须是一维非空数组: actual={indices.shape}")
	if np.any(indices < 0) or np.any(indices >= contact_positions.shape[1]):
		raise ValueError(
			f"reference_indices 越界: contact_count={contact_positions.shape[1]}, "
			f"actual={indices.tolist()}"
		)
	speed_threshold_mps = float(speed_threshold_mps)
	inlier_tolerance_m = float(inlier_tolerance_m)
	minimum_samples = int(minimum_samples)
	if not np.isfinite(speed_threshold_mps) or speed_threshold_mps <= 0.0:
		raise ValueError(
			"稳定支撑速度阈值必须为正有限值: "
			f"actual={speed_threshold_mps}"
		)
	if not np.isfinite(inlier_tolerance_m) or inlier_tolerance_m <= 0.0:
		raise ValueError(
			"稳健地板内点容差必须为正有限值: "
			f"actual={inlier_tolerance_m}"
		)
	if minimum_samples < 3:
		raise ValueError(
			f"稳健地板最少样本数必须不小于 3: actual={minimum_samples}"
		)

	reference_heights = np.asarray(
		contact_positions[:, indices, 2], dtype=np.float64
	)
	reference_speeds = np.asarray(contact_speeds[:, indices], dtype=np.float64)
	finite = np.isfinite(reference_heights) & np.isfinite(reference_speeds)
	stable = finite & (reference_speeds <= speed_threshold_mps)
	stable_heights = np.sort(reference_heights[stable])
	if stable_heights.size < minimum_samples:
		raise ValueError(
			"低速足点不足，无法稳健拟合恒定地板: "
			f"stable_samples={stable_heights.size}, minimum={minimum_samples}, "
			f"speed_threshold_mps={speed_threshold_mps}"
		)

	maximum_span = 2.0 * inlier_tolerance_m
	best_start = 0
	best_end = 0
	best_count = 0
	best_median = np.inf
	window_start = 0
	for window_end in range(stable_heights.size):
		while (
			stable_heights[window_end] - stable_heights[window_start]
			> maximum_span
		):
			window_start += 1
		count = window_end - window_start + 1
		# 高度已经排序，可直接由两个中间下标得到窗口中位数，保持整个滑窗
		# 扫描为 O(N)，避免长动作对每个窗口重复排序/复制。
		lower_middle = (window_start + window_end) // 2
		upper_middle = (window_start + window_end + 1) // 2
		median = float(
			0.5
			* (
				stable_heights[lower_middle]
				+ stable_heights[upper_middle]
			)
		)
		if count > best_count or (count == best_count and median < best_median):
			best_start = window_start
			best_end = window_end + 1
			best_count = count
			best_median = median

	initial_cluster = stable_heights[best_start:best_end]
	if initial_cluster.size < minimum_samples:
		raise ValueError(
			"低速足点没有形成足够密集的地板簇: "
			f"cluster_samples={initial_cluster.size}, minimum={minimum_samples}, "
			f"inlier_tolerance_m={inlier_tolerance_m}"
		)
	initial_center = float(np.median(initial_cluster))
	inliers = stable_heights[
		np.abs(stable_heights - initial_center) <= inlier_tolerance_m
	]
	if inliers.size < minimum_samples:
		raise ValueError(
			"稳健地板重拟合后的内点不足: "
			f"inliers={inliers.size}, minimum={minimum_samples}"
		)
	floor_height = float(np.median(inliers))
	mad = float(np.median(np.abs(inliers - floor_height)))
	report: dict[str, float | int | str] = {
		"method": "stable_support_dense_median",
		"floor_height_m": floor_height,
		"speed_threshold_mps": speed_threshold_mps,
		"inlier_tolerance_m": inlier_tolerance_m,
		"minimum_samples": minimum_samples,
		"total_reference_samples": int(reference_heights.size),
		"stable_sample_count": int(stable_heights.size),
		"inlier_sample_count": int(inliers.size),
		"inlier_fraction_of_stable": float(inliers.size / stable_heights.size),
		"inlier_mad_m": mad,
		"stable_height_min_m": float(stable_heights[0]),
		"stable_height_max_m": float(stable_heights[-1]),
	}
	return floor_height, report



def resolve_contact_height_reference_indices(
	keypoint_names: list[str],
	contact_names: tuple[str, ...],
) -> np.ndarray:
	keypoint_index = {name: idx for idx, name in enumerate(keypoint_names)}
	resolved_indices = np.full(len(contact_names), -1, dtype=np.int32)
	for contact_idx, contact_name in enumerate(contact_names):
		candidate_names = (contact_name, f"{contact_name}_link")
		for candidate_name in candidate_names:
			if candidate_name in keypoint_index:
				resolved_indices[contact_idx] = keypoint_index[candidate_name]
				break
	return resolved_indices


def apply_low_pass_filter(values: np.ndarray, alpha: float) -> np.ndarray:
	if values.ndim != 1:
		raise ValueError(f"values shape must be [T], got {values.shape}")
	if not (0.0 < float(alpha) <= 1.0):
		raise ValueError(f"Low-pass alpha must be in (0, 1], got {alpha}")

	filtered = values.astype(np.float32).copy()
	for frame_idx in range(1, filtered.shape[0]):
		filtered[frame_idx] = (
			float(alpha) * filtered[frame_idx]
			+ (1.0 - float(alpha)) * filtered[frame_idx - 1]
		)
	return filtered


def compute_contact_height_offsets(
	keypoints: np.ndarray,
	keypoint_names: list[str],
	contact_names: tuple[str, ...],
	contact_positions: np.ndarray,
	contact_states: np.ndarray,
	height_lpf_alpha: float = 1.0,
	prevent_penetration: bool = False,
	initial_height: float = 0.0,
) -> np.ndarray:
	if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
		raise ValueError(f"keypoints shape must be [T, K, 3], got {keypoints.shape}")
	if contact_positions.ndim != 3 or contact_positions.shape[-1] != 3:
		raise ValueError(
			"contact_positions shape must be [T, C, 3], got "
			f"{contact_positions.shape}"
		)
	if contact_states.shape != contact_positions.shape[:2]:
		raise ValueError(
			"contact_states shape must match contact_positions[:2], got "
			f"{contact_states.shape} and {contact_positions.shape[:2]}"
		)
	if keypoints.shape[0] != contact_positions.shape[0]:
		raise ValueError(
			"keypoints and contact_positions must have the same frame count, got "
			f"{keypoints.shape[0]} and {contact_positions.shape[0]}"
		)
	initial_height = float(initial_height)
	if not np.isfinite(initial_height):
		raise ValueError(f"initial_height must be finite, got {initial_height}")

	contact_keypoint_indices = resolve_contact_height_reference_indices(keypoint_names, contact_names)
	height_offsets = np.zeros(contact_positions.shape[0], dtype=np.float32)
	last_height = initial_height
	for frame_idx in range(contact_positions.shape[0]):
		active_mask = np.asarray(contact_states[frame_idx], dtype=np.bool_)
		if not np.any(active_mask):
			height_offsets[frame_idx] = last_height
			continue

		active_contact_indices = np.flatnonzero(active_mask)
		active_heights: list[float] = []
		for contact_idx in active_contact_indices:
			keypoint_idx = int(contact_keypoint_indices[contact_idx])
			if keypoint_idx >= 0:
				active_heights.append(float(keypoints[frame_idx, keypoint_idx, 2]))
			else:
				active_heights.append(float(contact_positions[frame_idx, contact_idx, 2]))

		last_height = min(active_heights)
		height_offsets[frame_idx] = last_height
	if height_offsets.shape[0] <= 1 or float(height_lpf_alpha) >= 1.0:
		return height_offsets
	filtered = apply_low_pass_filter(height_offsets, alpha=height_lpf_alpha)
	if prevent_penetration:
		# A causal low-pass can lag when the floor estimate moves down. Clamping the
		# filtered value to the current raw minimum keeps active reference contacts
		# at or above z=0 without changing the contact detector or its thresholds.
		filtered = np.minimum(filtered, height_offsets)
	return filtered


def offset_keypoints_by_contact_height(
	keypoints: np.ndarray,
	keypoint_names: list[str],
	contact_names: tuple[str, ...],
	contact_positions: np.ndarray,
	contact_states: np.ndarray,
	height_lpf_alpha: float = 1.0,
	prevent_penetration: bool = False,
	initial_height: float = 0.0,
	dynamic_offset_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
	"""按接触高度移动整组关键点，或只应用全序列恒定高度标定。

	默认值保持仓库原有机器人行为。配置关闭动态偏移时，不读取逐帧接触状态，
	而是把 ``initial_height`` 作为整段唯一的 Z 偏移量；因此接触状态如何切换都
	不会改变全身高度。若 ``initial_height`` 为零，则关键点保持完全不变。
	"""
	if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
		raise ValueError(f"keypoints shape must be [T, K, 3], got {keypoints.shape}")
	initial_height = float(initial_height)
	if not np.isfinite(initial_height):
		raise ValueError(f"initial_height must be finite, got {initial_height}")
	if not bool(dynamic_offset_enabled):
		height_offsets = np.full(
			keypoints.shape[0], initial_height, dtype=np.float32
		)
		adjusted_keypoints = keypoints.copy()
		adjusted_keypoints[:, :, 2] -= height_offsets[:, None]
		return adjusted_keypoints.astype(np.float32), height_offsets
	height_offsets = compute_contact_height_offsets(
		keypoints=keypoints,
		keypoint_names=keypoint_names,
		contact_names=contact_names,
		contact_positions=contact_positions,
		contact_states=contact_states,
		height_lpf_alpha=height_lpf_alpha,
		prevent_penetration=prevent_penetration,
		initial_height=initial_height,
	)
	adjusted_keypoints = keypoints.copy()
	adjusted_keypoints[:, :, 2] -= height_offsets[:, None]
	return adjusted_keypoints.astype(np.float32), height_offsets


def append_robot_foot_keypoints(
	keypoints: np.ndarray,
	keypoint_quaternions: np.ndarray,
	robot_links: dict[str, tuple[str, str]],
	robot_config_path: Path,
	robot_mjcf_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
	target_contact_links = load_yaml_body_list_config(robot_config_path, "contact_links")
	contact_name_to_body = {
		canonicalize_contact_name(body_name): body_name for body_name in target_contact_links
	}
	anchor_specs = (
		("left_foot_end", "left_calf"),
		("right_foot_end", "right_calf"),
		("left_toe", "left_calf"),
		("right_toe", "right_calf"),
	)
	keypoint_index = {name: idx for idx, name in enumerate(["hips_mean", *robot_links.keys()])}
	extra_positions: list[np.ndarray] = []
	extra_quaternions: list[np.ndarray] = []
	extra_names: list[str] = []

	for contact_name, anchor_link_name in anchor_specs:
		target_body_name = contact_name_to_body.get(contact_name)
		if target_body_name is None:
			continue
		if anchor_link_name not in robot_links:
			raise ValueError(f"Missing robot link required for extra foot keypoints: {anchor_link_name}")
		anchor_body_name = robot_links[anchor_link_name][1]
		anchor_idx = keypoint_index[anchor_link_name]
		anchor_positions = keypoints[:, anchor_idx, :]
		anchor_quaternions = keypoint_quaternions[:, anchor_idx, :]
		local_offset = compute_robot_body_local_offset(
			robot_mjcf_path,
			anchor_body=anchor_body_name,
			target_body=target_body_name,
		)
		world_offset = quat_rotate_vectors_wxyz(anchor_quaternions, local_offset)
		extra_positions.append(anchor_positions + world_offset)
		extra_quaternions.append(anchor_quaternions)
		extra_names.append(target_body_name)

	if not extra_positions:
		return keypoints, keypoint_quaternions, extra_names

	extra_positions_arr = np.stack(extra_positions, axis=1)
	extra_quaternions_arr = np.stack(extra_quaternions, axis=1)
	return (
		np.concatenate([keypoints, extra_positions_arr], axis=1),
		np.concatenate([keypoint_quaternions, extra_quaternions_arr], axis=1),
		extra_names,
	)


def compute_link_geometry_from_positions(
	link_pairs: dict[str, tuple[str, str]],
	body_slots: dict[str, int],
	positions: np.ndarray,
) -> tuple[
		dict[str, np.ndarray],
		dict[str, np.ndarray],
		dict[str, tuple[str, str]],
]:
	link_lengths: dict[str, np.ndarray] = {}
	link_vectors: dict[str, np.ndarray] = {}
	resolved_pairs: dict[str, tuple[str, str]] = {}

	for link_name, (parent_body, child_body) in link_pairs.items():
		if parent_body not in body_slots:
			raise ValueError(f"Missing link parent body in replay buffers: {parent_body}")
		if child_body not in body_slots:
			raise ValueError(f"Missing link child body in replay buffers: {child_body}")

		parent_idx = body_slots[parent_body]
		child_idx = body_slots[child_body]
		link_vector = positions[:, child_idx, :] - positions[:, parent_idx, :]
		link_length = np.linalg.norm(link_vector, axis=-1)
		if np.any(link_length <= 1e-8):
			raise ValueError(f"Skeleton link has near-zero length for at least one frame: {link_name}")

		link_lengths[link_name] = link_length.astype(np.float32)
		link_vectors[link_name] = link_vector.astype(np.float32)
		resolved_pairs[link_name] = (parent_body, child_body)
		# link_is_static[link_name] = bool(np.allclose(link_length, link_length[0], rtol=1e-5, atol=1e-6))

	return link_lengths, link_vectors, resolved_pairs


def compute_link_scale_factors(
	robot_link_lengths: dict[str, float],
	skeleton_link_lengths: dict[str, np.ndarray],
) -> tuple[dict[str, float | np.ndarray], dict[str, bool]]:
	link_scales: dict[str, float | np.ndarray] = {}
	link_scale_is_static: dict[str, bool] = {}

	for link_name, skeleton_length in skeleton_link_lengths.items():
		if link_name not in robot_link_lengths:
			raise ValueError(f"Missing robot link length for skeleton link: {link_name}")

		scale = robot_link_lengths[link_name] / skeleton_length
		link_scales[link_name] = scale.astype(np.float32)
		link_scale_is_static[link_name] = False

	return link_scales, link_scale_is_static


def compute_leg_displacement_scale(
	robot_link_lengths: dict[str, float],
	skeleton_link_vectors: dict[str, np.ndarray],
	knee_angle_offset_degrees: float,
) -> tuple[float, float, float]:
	required_robot_links = ("left_thigh", "right_thigh", "left_calf", "right_calf")
	for link_name in required_robot_links:
		if link_name not in robot_link_lengths:
			raise ValueError(f"Missing robot leg link length: {link_name}")

	required_skeleton_links = ("left_thigh", "right_thigh", "left_calf", "right_calf")
	for link_name in required_skeleton_links:
		if link_name not in skeleton_link_vectors:
			raise ValueError(f"Missing skeleton leg link vector: {link_name}")

	robot_thigh_length = 0.5 * (
		float(robot_link_lengths["left_thigh"]) + float(robot_link_lengths["right_thigh"])
	)
	robot_calf_length = 0.5 * (
		float(robot_link_lengths["left_calf"]) + float(robot_link_lengths["right_calf"])
	)
	robot_leg_length = float(
		np.sqrt(
			max(
				robot_thigh_length**2
				+ robot_calf_length**2
				+ 2.0
				* robot_thigh_length
				* robot_calf_length
				* np.cos(np.radians(float(knee_angle_offset_degrees))),
				0.0,
			)
		)
	)

	left_skeleton_leg_length = np.linalg.norm(skeleton_link_vectors["left_thigh"], axis=-1) + np.linalg.norm(
		skeleton_link_vectors["left_calf"], axis=-1
	)
	right_skeleton_leg_length = np.linalg.norm(skeleton_link_vectors["right_thigh"], axis=-1) + np.linalg.norm(
		skeleton_link_vectors["right_calf"], axis=-1
	)
	skeleton_leg_length = float(np.mean(0.5 * (left_skeleton_leg_length + right_skeleton_leg_length)))
	if skeleton_leg_length <= 1e-8:
		raise ValueError(f"Skeleton leg length must be positive, got {skeleton_leg_length}")

	leg_displacement_scale = float(robot_leg_length / skeleton_leg_length)
	return leg_displacement_scale, robot_leg_length, skeleton_leg_length


def scale_keypoint_frame_displacements(
	keypoints: np.ndarray,
	displacement_scale: float,
	root_keypoint_idx: int = 0,
) -> np.ndarray:
	if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
		raise ValueError(f"keypoints shape must be [T, K, 3], got {keypoints.shape}")
	if displacement_scale <= 0.0:
		raise ValueError(f"displacement_scale must be positive, got {displacement_scale}")
	if root_keypoint_idx < 0 or root_keypoint_idx >= keypoints.shape[1]:
		raise ValueError(
			f"root_keypoint_idx must be within [0, {keypoints.shape[1]}), got {root_keypoint_idx}"
		)
	if keypoints.shape[0] <= 1 or np.isclose(displacement_scale, 1.0):
		return keypoints.astype(np.float32, copy=True)

	root_positions = keypoints[:, root_keypoint_idx, :].astype(np.float32)
	root_frame_deltas = np.diff(root_positions, axis=0)
	scaled_root_positions = np.empty_like(root_positions, dtype=np.float32)
	scaled_root_positions[0] = root_positions[0]
	scaled_root_positions[1:] = scaled_root_positions[0:1] + np.cumsum(
		(displacement_scale * root_frame_deltas).astype(np.float32),
		axis=0,
	)
	translation_offsets = scaled_root_positions - root_positions
	return (keypoints + translation_offsets[:, None, :]).astype(np.float32)


def apply_link_scales_to_positions(
	positions: np.ndarray,
	link_pairs: dict[str, tuple[str, str]],
	body_slots: dict[str, int],
	link_scales: dict[str, float | np.ndarray],
	link_vectors: dict[str, np.ndarray],
	knee_angle_offset_degrees: float,
	left_leg_links: tuple[str, str, str],
	right_leg_links: tuple[str, str, str],
	quaternions: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
	scaled_positions = positions.copy()
	scaled_quaternions = quaternions.copy() if quaternions is not None else None
	for link_name, (parent_body, child_body) in link_pairs.items():
		parent_idx = body_slots[parent_body]
		child_idx = body_slots[child_body]
		scale = link_scales[link_name]
		if np.isscalar(scale):
			scaled_positions[:, child_idx, :] = (
				scaled_positions[:, parent_idx, :] + float(scale) * link_vectors[link_name]
			)
		else:
			scaled_positions[:, child_idx, :] = (
				scaled_positions[:, parent_idx, :] + scale[:, None] * link_vectors[link_name]
			)

	# Blend knee joint by increasing knee angle and only updating foot keypoints.
	blend_offset_radius = np.radians(float(knee_angle_offset_degrees))

	for body_name in left_leg_links + right_leg_links:
		if body_name not in body_slots:
			raise ValueError(f"Missing leg body in replay buffers: {body_name}")

	left_hip_idx = body_slots[left_leg_links[0]]
	left_knee_idx = body_slots[left_leg_links[1]]
	left_foot_idx = body_slots[left_leg_links[2]]
	left_hip_to_knee = scaled_positions[:, left_knee_idx, :] - scaled_positions[:, left_hip_idx, :]
	left_knee_to_foot = scaled_positions[:, left_foot_idx, :] - scaled_positions[:, left_knee_idx, :]
	left_hip_to_knee_length = np.linalg.norm(left_hip_to_knee, axis=-1)
	left_knee_to_foot_length = np.linalg.norm(left_knee_to_foot, axis=-1)

	right_hip_idx = body_slots[right_leg_links[0]]
	right_knee_idx = body_slots[right_leg_links[1]]
	right_foot_idx = body_slots[right_leg_links[2]]
	right_hip_to_knee = scaled_positions[:, right_knee_idx, :] - scaled_positions[:, right_hip_idx, :]
	right_knee_to_foot = scaled_positions[:, right_foot_idx, :] - scaled_positions[:, right_knee_idx, :]
	right_hip_to_knee_length = np.linalg.norm(right_hip_to_knee, axis=-1)
	right_knee_to_foot_length = np.linalg.norm(right_knee_to_foot, axis=-1)

	left_hip_to_foot = scaled_positions[:, left_foot_idx, :] - scaled_positions[:, left_hip_idx, :]
	right_hip_to_foot = scaled_positions[:, right_foot_idx, :] - scaled_positions[:, right_hip_idx, :]
	left_hip_to_foot_length = np.linalg.norm(left_hip_to_foot, axis=-1)
	right_hip_to_foot_length = np.linalg.norm(right_hip_to_foot, axis=-1)
	left_hip_to_foot_dir_norm = left_hip_to_foot / np.clip(left_hip_to_foot_length[:, None], 1e-8, None)
	right_hip_to_foot_dir_norm = right_hip_to_foot / np.clip(right_hip_to_foot_length[:, None], 1e-8, None)

	left_cos = np.sum(left_hip_to_knee * left_knee_to_foot, axis=-1) / np.clip(
		left_hip_to_knee_length * left_knee_to_foot_length,
		1e-8,
		None,
	)
	right_cos = np.sum(right_hip_to_knee * right_knee_to_foot, axis=-1) / np.clip(
		right_hip_to_knee_length * right_knee_to_foot_length,
		1e-8,
		None,
	)
	current_angle_left = np.arccos(np.clip(left_cos, -1.0, 1.0))
	blended_angle_left = current_angle_left + blend_offset_radius
	current_angle_right = np.arccos(np.clip(right_cos, -1.0, 1.0))
	blended_angle_right = current_angle_right + blend_offset_radius

	left_hip_to_foot_blended_length = np.sqrt(
		np.clip(
			left_hip_to_knee_length**2
			+ left_knee_to_foot_length**2
			+ 2 * left_hip_to_knee_length * left_knee_to_foot_length * np.cos(blended_angle_left),
			0.0,
			None,
		)
	)
	right_hip_to_foot_blended_length = np.sqrt(
		np.clip(
			right_hip_to_knee_length**2
			+ right_knee_to_foot_length**2
			+ 2 * right_hip_to_knee_length * right_knee_to_foot_length * np.cos(blended_angle_right),
			0.0,
			None,
		)
	)

	left_blended_foot_pos = (
		scaled_positions[:, left_hip_idx, :]
		+ left_hip_to_foot_dir_norm * left_hip_to_foot_blended_length[:, None]
	)
	right_blended_foot_pos = (
		scaled_positions[:, right_hip_idx, :]
		+ right_hip_to_foot_dir_norm * right_hip_to_foot_blended_length[:, None]
	)
	left_blended_knee_pos = compute_two_bone_ik_knee_positions(
		hip_positions=scaled_positions[:, left_hip_idx, :],
		knee_positions=scaled_positions[:, left_knee_idx, :],
		foot_positions=scaled_positions[:, left_foot_idx, :],
		target_foot_positions=left_blended_foot_pos,
	)
	right_blended_knee_pos = compute_two_bone_ik_knee_positions(
		hip_positions=scaled_positions[:, right_hip_idx, :],
		knee_positions=scaled_positions[:, right_knee_idx, :],
		foot_positions=scaled_positions[:, right_foot_idx, :],
		target_foot_positions=right_blended_foot_pos,
	)

	scaled_positions[:, left_foot_idx, :] = left_blended_foot_pos
	scaled_positions[:, right_foot_idx, :] = right_blended_foot_pos
	scaled_positions[:, left_knee_idx, :] = left_blended_knee_pos
	scaled_positions[:, right_knee_idx, :] = right_blended_knee_pos

	if scaled_quaternions is not None:
		# Rotate the hip and knee orientations to follow the bent leg, while keeping
		# the foot orientation untouched. The hip body follows the thigh segment
		# (hip -> knee) and the knee body follows the shank segment (knee -> foot).
		# Each rotation is about the normal of the hip/knee/foot plane (the cross
		# product of the original and blended segment directions).
		left_hip_to_knee_blended = left_blended_knee_pos - scaled_positions[:, left_hip_idx, :]
		left_knee_to_foot_blended = left_blended_foot_pos - left_blended_knee_pos
		right_hip_to_knee_blended = right_blended_knee_pos - scaled_positions[:, right_hip_idx, :]
		right_knee_to_foot_blended = right_blended_foot_pos - right_blended_knee_pos

		left_hip_delta = batch_rotation_between_vectors_wxyz(left_hip_to_knee, left_hip_to_knee_blended)
		left_knee_delta = batch_rotation_between_vectors_wxyz(left_knee_to_foot, left_knee_to_foot_blended)
		right_hip_delta = batch_rotation_between_vectors_wxyz(right_hip_to_knee, right_hip_to_knee_blended)
		right_knee_delta = batch_rotation_between_vectors_wxyz(right_knee_to_foot, right_knee_to_foot_blended)

		scaled_quaternions[:, left_hip_idx, :] = multiply_quaternions_wxyz(
			left_hip_delta, scaled_quaternions[:, left_hip_idx, :]
		)
		scaled_quaternions[:, left_knee_idx, :] = multiply_quaternions_wxyz(
			left_knee_delta, scaled_quaternions[:, left_knee_idx, :]
		)
		scaled_quaternions[:, right_hip_idx, :] = multiply_quaternions_wxyz(
			right_hip_delta, scaled_quaternions[:, right_hip_idx, :]
		)
		scaled_quaternions[:, right_knee_idx, :] = multiply_quaternions_wxyz(
			right_knee_delta, scaled_quaternions[:, right_knee_idx, :]
		)
		return scaled_positions, scaled_quaternions

	return scaled_positions


def build_retarget_keypoints(
	skeleton_positions: np.ndarray,
	skeleton_quaternions: np.ndarray,
	robot_link_lengths: dict[str, float],
	robot_config_path: Path,
	skeleton_config_path: Path,
) -> tuple[
		np.ndarray,
		np.ndarray,
		dict[str, np.ndarray],
		dict[str, float | np.ndarray],
		dict[str, bool],
	]:
	body_slots = {name: idx for idx, name in enumerate(REPLAY_BODY_NAMES)}
	robot_links = load_robot_links_config(robot_config_path)
	skeleton_links = load_skeleton_links_config(skeleton_config_path)
	expected_links = set(skeleton_links)
	actual_links = set(robot_links)
	if expected_links != actual_links:
		raise ValueError(
			"robot_links 与 skeleton_links 必须完整一致: "
			f"robot_config={robot_config_path}, skeleton_config={skeleton_config_path}, "
			f"missing={sorted(expected_links-actual_links)}, extra={sorted(actual_links-expected_links)}"
		)
	for link_name, length in robot_link_lengths.items():
		if not np.isfinite(length) or length <= 1.0e-6:
			raise ValueError(
				f"机器人连杆长度必须为正有限值: path={robot_config_path}, link={link_name}, actual={length}"
			)
	for left_name, right_name in (
		("left_thigh", "right_thigh"),
		("left_calf", "right_calf"),
		("left_arm", "right_arm"),
		("left_fore_arm", "right_fore_arm"),
	):
		left_length = robot_link_lengths[left_name]
		right_length = robot_link_lengths[right_name]
		relative_difference = abs(left_length - right_length) / max(left_length, right_length)
		if relative_difference > 0.15:
			raise ValueError(
				f"左右对称连杆长度差超过 15%: left={left_name}:{left_length}, "
				f"right={right_name}:{right_length}, relative={relative_difference}"
			)
		if relative_difference > 0.05:
			print(
				f"[warning] 左右连杆长度差超过 5%: {left_name}/{right_name}={relative_difference:.3%}"
			)
	knee_angle_offset_degrees = load_scalar_float_config(
		robot_config_path,
		"knee_angle_offset_degrees",
		default=30.0,
	)
	key_frame_offset_degrees, key_frame_axis_map = load_key_frame_config(
		robot_config_path,
		section_name="key_frame_config",
	)
	left_leg_links = derive_leg_body_chain_from_links(skeleton_links, "left")
	right_leg_links = derive_leg_body_chain_from_links(skeleton_links, "right")
	skeleton_link_lengths, skeleton_link_vectors, resolved_skeleton_links = compute_link_geometry_from_positions(
		skeleton_links,
		body_slots,
		skeleton_positions,
	)
	link_scales, link_scale_is_static = compute_link_scale_factors(
		robot_link_lengths,
		skeleton_link_lengths,
	)
	retarget_keypoints, retarget_quaternions = apply_link_scales_to_positions(
		skeleton_positions,
		resolved_skeleton_links,
		body_slots,
		link_scales,
		skeleton_link_vectors,
		knee_angle_offset_degrees,
		left_leg_links,
		right_leg_links,
		quaternions=skeleton_quaternions,
	)
	ordered_keypoints = np.zeros((skeleton_positions.shape[0], len(robot_links) + 1, 3), dtype=np.float32)
	ordered_keypoint_quaternions = np.zeros((skeleton_positions.shape[0], len(robot_links) + 1, 4), dtype=np.float32)
	ordered_keypoints[:, 0, :] = retarget_keypoints[:, body_slots["hips_mean"], :]
	hips_mean_offset = key_frame_offset_degrees.get("hips_mean", np.zeros(3, dtype=np.float32))
	hips_mean_axis_map = key_frame_axis_map.get("hips_mean", np.eye(3, dtype=np.float32))
	ordered_keypoint_quaternions[:, 0, :] = apply_axis_map_and_local_euler_offset_wxyz(
		retarget_quaternions[:, body_slots["hips_mean"], :],
		hips_mean_axis_map,
		hips_mean_offset,
	)
	for keypoint_idx, link_name in enumerate(robot_links, start=1):
		if link_name not in resolved_skeleton_links:
			raise ValueError(f"Missing skeleton link for robot link: {link_name}")
		_parent_body, child_body = resolved_skeleton_links[link_name]
		child_idx = body_slots[child_body]
		ordered_keypoints[:, keypoint_idx, :] = retarget_keypoints[:, child_idx, :]
		euler_offset = key_frame_offset_degrees.get(child_body, np.zeros(3, dtype=np.float32))
		axis_map = key_frame_axis_map.get(child_body, np.eye(3, dtype=np.float32))
		ordered_keypoint_quaternions[:, keypoint_idx, :] = apply_axis_map_and_local_euler_offset_wxyz(
			retarget_quaternions[:, child_idx, :],
			axis_map,
			euler_offset,
		)

	return (
		ordered_keypoints,
		ordered_keypoint_quaternions,
		skeleton_link_vectors,
		link_scales,
		link_scale_is_static,
	)


def update_viewer_keypoints(
	viewer,
	keypoints: np.ndarray,
	keypoint_quaternions: np.ndarray | None = None,
	rgba: np.ndarray = RETARGET_KEYPOINT_RGBA,
	radius: float = RETARGET_KEYPOINT_RADIUS,
	overlay_keypoints: np.ndarray | None = None,
	overlay_rgba: np.ndarray = RETARGET_BLEND_FOOT_RGBA,
	overlay_radius: float = RETARGET_BLEND_FOOT_RADIUS,
	knee_overlay_keypoints: np.ndarray | None = None,
	knee_overlay_rgba: np.ndarray = RETARGET_BLEND_KNEE_RGBA,
	knee_overlay_radius: float = RETARGET_BLEND_KNEE_RADIUS,
	contact_keypoints: np.ndarray | None = None,
	contact_states: np.ndarray | None = None,
	contact_rgba: np.ndarray = CONTACT_ACTIVE_RGBA,
	contact_radius: float = CONTACT_ACTIVE_RADIUS,
) -> None:
	scene = viewer.user_scn
	max_geoms = len(scene.geoms)
	total_keypoints = keypoints.shape[0]
	axis_geoms = 0
	active_contact_keypoints = None
	if keypoint_quaternions is not None:
		if keypoint_quaternions.shape != (keypoints.shape[0], 4):
			raise ValueError(
				"keypoint_quaternions shape must be [N, 4], got "
				f"{keypoint_quaternions.shape} for N={keypoints.shape[0]}"
			)
		axis_geoms = 3 * keypoints.shape[0]
	if overlay_keypoints is not None:
		total_keypoints += overlay_keypoints.shape[0]
	if knee_overlay_keypoints is not None:
		total_keypoints += knee_overlay_keypoints.shape[0]
	if contact_keypoints is not None:
		if contact_states is None:
			raise ValueError("contact_states must be provided when contact_keypoints are provided")
		if contact_keypoints.shape[0] != contact_states.shape[0]:
			raise ValueError(
				"contact_states length must match contact_keypoints, got "
				f"{contact_states.shape[0]} and {contact_keypoints.shape[0]}"
			)
		active_contact_keypoints = contact_keypoints[np.asarray(contact_states, dtype=np.bool_)]
		total_keypoints += active_contact_keypoints.shape[0]
	total_geoms = total_keypoints + axis_geoms
	if total_geoms > max_geoms:
		raise ValueError(f"Too many keypoint geoms for viewer scene: {total_geoms} > {max_geoms}")

	identity_mat = np.eye(3, dtype=np.float32).reshape(-1)
	scene.ngeom = 0
	translation_offset = np.array((1.0, 0.0, 0.0), dtype=np.float32)
	contact_translation_offset = np.zeros(3, dtype=np.float32)
	for idx, point in enumerate(keypoints):
		mujoco.mjv_initGeom(
			scene.geoms[idx],
			mujoco.mjtGeom.mjGEOM_SPHERE,
			np.array([radius, radius, radius], dtype=np.float32),
			point.astype(np.float32)+np.array(translation_offset, dtype=np.float32),
			identity_mat,
			rgba,
		)
		scene.ngeom += 1

	if keypoint_quaternions is not None:
		for idx, (point, quat_wxyz) in enumerate(zip(keypoints, keypoint_quaternions)):
			mat_flat = np.zeros(9, dtype=np.float64)
			mujoco.mju_quat2Mat(mat_flat, quat_wxyz.astype(np.float64))
			rz = mat_flat.reshape(3, 3).astype(np.float32)
			ry = rz @ ROT_X_NEG_90
			rx = rz @ ROT_Y_POS_90
			axes_rot = [rx, ry, rz]
			for axis_idx, axis_rot in enumerate(axes_rot):
				mujoco.mjv_initGeom(
					scene.geoms[scene.ngeom],
					mujoco.mjtGeom.mjGEOM_CYLINDER,
					np.array(
						[
							RETARGET_KEYPOINT_AXIS_RADIUS,
							RETARGET_KEYPOINT_AXIS_HALF_LENGTH,
							0.0,
						],
						dtype=np.float32,
					),
					point.astype(np.float32)
					+ translation_offset
					+ axis_rot @ np.array([0.0, 0.0, 0.5 * RETARGET_KEYPOINT_AXIS_HALF_LENGTH], dtype=np.float32),
					axis_rot.reshape(-1),
					RETARGET_KEYPOINT_AXIS_COLORS[axis_idx],
				)
				scene.ngeom += 1

	if overlay_keypoints is not None:
		for idx, point in enumerate(overlay_keypoints):
			mujoco.mjv_initGeom(
				scene.geoms[scene.ngeom],
				mujoco.mjtGeom.mjGEOM_SPHERE,
				np.array([overlay_radius, overlay_radius, overlay_radius], dtype=np.float32),
				point.astype(np.float32) + np.array(translation_offset, dtype=np.float32),
				identity_mat,
				overlay_rgba,
			)
			scene.ngeom += 1

	if knee_overlay_keypoints is not None:
		for idx, point in enumerate(knee_overlay_keypoints):
			mujoco.mjv_initGeom(
				scene.geoms[scene.ngeom],
				mujoco.mjtGeom.mjGEOM_SPHERE,
				np.array([knee_overlay_radius, knee_overlay_radius, knee_overlay_radius], dtype=np.float32),
				point.astype(np.float32) + np.array(translation_offset, dtype=np.float32),
				identity_mat,
				knee_overlay_rgba,
			)
			scene.ngeom += 1

	if active_contact_keypoints is not None:
		for point in active_contact_keypoints:
			mujoco.mjv_initGeom(
				scene.geoms[scene.ngeom],
				mujoco.mjtGeom.mjGEOM_SPHERE,
				np.array([contact_radius, contact_radius, contact_radius], dtype=np.float32),
				point.astype(np.float32) + contact_translation_offset,
				identity_mat,
				contact_rgba,
			)
			scene.ngeom += 1


def require_smplx() -> None:
	if importlib.util.find_spec("smplx") is None:
		raise ImportError(
			"Missing dependency 'smplx'. Install it in the mjlab environment with: "
			"python -m pip install smplx"
		)


def normalize_gender(raw_gender: object, override: str) -> str:
	if override != "auto":
		return override

	gender = str(np.asarray(raw_gender).item()).strip().lower()
	if gender in {"female", "male", "neutral"}:
		return gender
	return "neutral"


def _convert_y_up_to_z_up(
	trans: np.ndarray, root_orient: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
	"""Rotate a Y-up motion (camera/video world) into the Z-up world MuJoCo uses.

	Some video-derived SMPL-X exports store the world with +Y as the up axis
	(gravity along -Y). MuJoCo and the rest of this pipeline assume +Z up. We
	apply a fixed +90 deg rotation about the world X axis to both the root
	translation and the root orientation so the body stands upright.
	"""
	from scipy.spatial.transform import Rotation as _R

	# +90 deg about world X maps world +Y -> +Z.
	world_fix = _R.from_euler("x", 90.0, degrees=True)
	fix_mat = world_fix.as_matrix().astype(np.float32)

	trans_z = trans @ fix_mat.T
	root_rot = _R.from_rotvec(root_orient)
	root_orient_z = (world_fix * root_rot).as_rotvec().astype(np.float32)
	return trans_z.astype(np.float32), root_orient_z


def _is_y_up(trans: np.ndarray, root_orient: np.ndarray) -> bool:
	"""Heuristically decide whether the motion is stored Y-up instead of Z-up.

	Checks where the body's local up axis (+Y in SMPL) points in the world for
	the first frame. If it aligns with world +Y more than world +Z, the motion
	is Y-up.
	"""
	from scipy.spatial.transform import Rotation as _R

	mat = _R.from_rotvec(root_orient[0]).as_matrix()
	local_up_world = mat[:, 1]  # world direction of body local +Y (up)
	return abs(local_up_world[1]) > abs(local_up_world[2])


def _first_present(data: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> str | None:
	for name in names:
		if name in data.files:
			return name
	return None


def _scalar_string(data: np.lib.npyio.NpzFile, name: str, default: str) -> str:
	if name not in data.files:
		return default
	return str(np.asarray(data[name]).reshape(-1)[0]).strip()


def _resample_rotvecs(
	values: np.ndarray,
	source_times: np.ndarray,
	target_times: np.ndarray,
	field_name: str,
) -> np.ndarray:
	if values.ndim != 2 or values.shape[1] % 3 != 0:
		raise ValueError(
			f"旋转向量字段必须为 [T,3*N]: field={field_name}, actual={values.shape}"
		)
	joint_count = values.shape[1] // 3
	result = np.empty((target_times.shape[0], values.shape[1]), dtype=np.float32)
	for joint_idx in range(joint_count):
		joint_values = values[:, 3 * joint_idx : 3 * (joint_idx + 1)]
		rotations = Rotation.from_rotvec(joint_values)
		result[:, 3 * joint_idx : 3 * (joint_idx + 1)] = Slerp(
			source_times, rotations
		)(target_times).as_rotvec().astype(np.float32)
	return result


def resample_motion_arrays(
	motion: dict[str, np.ndarray],
	source_fps: float,
	target_fps: float,
) -> dict[str, np.ndarray]:
	"""按真实时间重采样平移、关节旋转与可选表情数据。"""
	if not np.isfinite(source_fps) or source_fps <= 0.0:
		raise ValueError(f"源 fps 必须为正有限值: actual={source_fps}")
	if not np.isfinite(target_fps) or target_fps <= 0.0:
		raise ValueError(f"目标 fps 必须为正有限值: actual={target_fps}")
	num_frames = int(motion["trans"].shape[0])
	if num_frames <= 1 or np.isclose(source_fps, target_fps):
		return {key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value for key, value in motion.items()}
	duration = (num_frames - 1) / float(source_fps)
	source_times = np.arange(num_frames, dtype=np.float64) / float(source_fps)
	target_count = round(duration * float(target_fps)) + 1
	target_times = np.arange(target_count, dtype=np.float64) / float(target_fps)
	# round() 可能让末帧比源终点多不足半个目标周期；SLERP 不允许外插。
	target_times = np.minimum(target_times, source_times[-1])
	result = dict(motion)
	result["trans"] = np.column_stack(
		[np.interp(target_times, source_times, motion["trans"][:, axis]) for axis in range(3)]
	).astype(np.float32)
	result["root_orient"] = _resample_rotvecs(
		motion["root_orient"], source_times, target_times, "root_orient"
	)
	result["pose_body"] = _resample_rotvecs(
		motion["pose_body"], source_times, target_times, "pose_body"
	)
	for field_name in ("pose_hand", "pose_jaw", "pose_eye"):
		if field_name in motion:
			result[field_name] = _resample_rotvecs(
				motion[field_name], source_times, target_times, field_name
			)
	if "expression" in motion:
		expression = np.asarray(motion["expression"], dtype=np.float32)
		result["expression"] = np.column_stack(
			[np.interp(target_times, source_times, expression[:, axis]) for axis in range(expression.shape[1])]
		).astype(np.float32)
	return result


def load_motion_arrays(
	motion_file: Path,
	model_type_override: str = "auto",
	translation_scale: float = 1.0,
	up_axis: str = "auto",
	target_fps: float = 0.0,
	max_frames: int = 0,
) -> tuple[dict[str, np.ndarray], str, float]:
	"""将标准化、AMASS 与四集合音乐 SMPL NPZ 统一为仓库内部格式。"""
	motion_file = motion_file.expanduser().resolve()
	if not motion_file.is_file():
		raise FileNotFoundError(f"动作 NPZ 不存在: {motion_file}")
	if not np.isfinite(translation_scale) or translation_scale <= 0.0:
		raise ValueError(
			f"translation_scale 必须为正有限值: path={motion_file}, actual={translation_scale}"
		)
	if model_type_override not in {"auto", "smpl", "smplx"}:
		raise ValueError(f"model_type_override 无效: actual={model_type_override}")
	if up_axis not in {"auto", "y", "z"}:
		raise ValueError(f"up_axis 无效: actual={up_axis}")
	if target_fps < 0.0 or not np.isfinite(target_fps):
		raise ValueError(f"target_fps 必须是 0 或正有限值: actual={target_fps}")
	if max_frames < 0:
		raise ValueError(f"max_frames 必须非负: actual={max_frames}")

	with np.load(motion_file, allow_pickle=True) as data:
		metadata_model_type = _scalar_string(data, "surface_model_type", "").lower()
		if model_type_override == "auto":
			surface_model_type = metadata_model_type or "smpl"
		else:
			surface_model_type = model_type_override
		if surface_model_type not in {"smpl", "smplx"}:
			raise ValueError(
				f"人体模型类型仅支持 smpl/smplx: path={motion_file}, actual={surface_model_type}"
			)

		translation_field = _first_present(data, ("trans", "transl", "translations"))
		if translation_field is None:
			raise KeyError(
				f"动作缺少平移字段: path={motion_file}, expected=trans|transl|translations, actual={data.files}"
			)
		if "root_orient" in data.files and "pose_body" in data.files:
			root_orient = np.asarray(data["root_orient"], dtype=np.float32)
			pose_body = np.asarray(data["pose_body"], dtype=np.float32)
		elif "global_orient" in data.files and "body_pose" in data.files:
			root_orient = np.asarray(data["global_orient"], dtype=np.float32)
			pose_body = np.asarray(data["body_pose"], dtype=np.float32)
		else:
			pose_field = _first_present(data, ("poses", "pose"))
			if pose_field is None:
				raise KeyError(
					f"动作缺少姿态字段: path={motion_file}, expected=root_orient+pose_body|poses|pose, actual={data.files}"
				)
			poses = np.asarray(data[pose_field], dtype=np.float32)
			if poses.ndim != 2 or poses.shape[1] < 66:
				raise ValueError(
					f"姿态字段必须为 [T,D>=66]: path={motion_file}, field={pose_field}, actual={poses.shape}"
				)
			root_orient = poses[:, :3]
			pose_body = poses[:, 3:66]

		trans = np.asarray(data[translation_field], dtype=np.float32) * float(translation_scale)
		num_frames = int(trans.shape[0]) if trans.ndim >= 1 else 0
		if trans.shape != (num_frames, 3):
			raise ValueError(
				f"平移字段必须为 [T,3]: path={motion_file}, field={translation_field}, actual={trans.shape}"
			)
		if root_orient.shape != (num_frames, 3):
			raise ValueError(
				f"root_orient 必须为 [T,3]: path={motion_file}, actual={root_orient.shape}, T={num_frames}"
			)
		if pose_body.shape != (num_frames, 63):
			raise ValueError(
				f"pose_body 必须为 [T,63]: path={motion_file}, actual={pose_body.shape}, T={num_frames}"
			)
		if num_frames == 0:
			raise ValueError(f"动作不得为空: path={motion_file}")
		if not np.all(np.isfinite(trans)) or not np.all(np.isfinite(root_orient)) or not np.all(np.isfinite(pose_body)):
			raise ValueError(f"动作核心字段包含 NaN/Inf: path={motion_file}")

		if "betas" in data.files:
			betas_raw = np.asarray(data["betas"], dtype=np.float32)
			betas = betas_raw[0] if betas_raw.ndim == 2 else betas_raw.reshape(-1)
		else:
			betas = np.zeros(10, dtype=np.float32)
		if betas.shape[0] < 1 or not np.all(np.isfinite(betas)):
			raise ValueError(f"betas 无效: path={motion_file}, actual={betas.shape}")

		motion = {
			"surface_model_type": surface_model_type,
			"trans": trans,
			"root_orient": root_orient,
			"pose_body": pose_body,
			"betas": betas,
		}
		for field_name in ("pose_hand", "pose_jaw", "pose_eye", "expression"):
			if field_name in data.files:
				optional = np.asarray(data[field_name], dtype=np.float32)
				if optional.ndim != 2 or optional.shape[0] != num_frames:
					raise ValueError(
						f"可选字段首维必须为 T: path={motion_file}, field={field_name}, actual={optional.shape}, T={num_frames}"
					)
				motion[field_name] = optional

		gender = normalize_gender(
			data["gender"] if "gender" in data.files else np.asarray("neutral"),
			override="auto",
		)
		fps_field = _first_present(
			data,
			("mocap_frame_rate", "mocap_framerate", "fps", "frame_rate", "framerate"),
		)
		fps = float(np.asarray(data[fps_field]).reshape(-1)[0]) if fps_field else 30.0
		coordinate_system = _scalar_string(data, "coordinate_system", "")

	if not np.isfinite(fps) or fps <= 0.0:
		raise ValueError(f"源 fps 必须为正有限值: path={motion_file}, field={fps_field}, actual={fps}")
	motion["source_fps"] = float(fps)
	motion["source_motion_file"] = str(motion_file)
	motion["source_coordinate_system"] = coordinate_system
	convert_y_up = up_axis == "y" or (up_axis == "auto" and _is_y_up(trans, root_orient))
	if convert_y_up:
		motion["trans"], motion["root_orient"] = _convert_y_up_to_z_up(
			motion["trans"], motion["root_orient"]
		)
		print(f"[load_motion] detected/requested Y-up motion in {motion_file.name}; converted to Z-up.")
	actual_fps = float(target_fps) if target_fps > 0.0 else float(fps)
	if target_fps > 0.0:
		motion = resample_motion_arrays(motion, source_fps=fps, target_fps=target_fps)
	if max_frames > 0:
		frame_count_before_slice = int(motion["trans"].shape[0])
		for field_name, value in list(motion.items()):
			if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == frame_count_before_slice:
				motion[field_name] = value[:max_frames].copy()
	return motion, gender, actual_fps


def resolve_model_root(model_path: Path, model_type: str, gender: str) -> tuple[Path, str]:
	if model_path.is_file():
		suffix = model_path.suffix.lstrip(".")
		if not suffix:
			raise ValueError(f"SMPL model file must have a suffix: {model_path}")
		return model_path, suffix

	candidate_dirs = [model_path, model_path / model_type]
	candidate_exts = ["pkl", "npz"] if model_type == "smpl" else ["npz", "pkl"]
	expected_prefix = model_type.upper()
	expected_gender = gender.upper()

	for directory in candidate_dirs:
		for ext in candidate_exts:
			candidate = directory / f"{expected_prefix}_{expected_gender}.{ext}"
			if candidate.exists():
				# smplx.create() 会在目录参数后再次追加 model_type 子目录。
				# 这里传入已验证存在的具体模型文件，既兼容 ``body_models/smplx``
				# 集合布局，也兼容 ``asset/smplx`` 直接存放模型文件的布局。
				return candidate, ext

	searched = ", ".join(str(path) for path in candidate_dirs)
	raise FileNotFoundError(
		f"Could not find {expected_prefix}_{expected_gender}.{{npz,pkl}} under: {searched}"
	)


def build_smpl_model(
	model_path: Path,
	model_type: str,
	gender: str,
	num_betas: int,
	device: str,
):
	require_smplx()
	smplx = importlib.import_module("smplx")

	model_root, ext = resolve_model_root(model_path, model_type, gender)

	body_model = smplx.create(
		model_path=str(model_root),
		model_type=model_type,
		gender=gender,
		num_betas=num_betas,
		use_pca=False,
		ext=ext,
	)
	return body_model.to(device)


def compute_joint_positions(
	motion: dict[str, np.ndarray],
	body_model,
	device: str,
	chunk_size: int,
	progress_desc: str | None = None,
) -> np.ndarray:
	num_frames = motion["trans"].shape[0]
	num_betas = int(body_model.num_betas)
	betas = torch.as_tensor(motion["betas"][:num_betas], dtype=torch.float32, device=device)
	betas = betas.unsqueeze(0)

	trans = torch.as_tensor(motion["trans"], dtype=torch.float32, device=device)
	root_orient = torch.as_tensor(motion["root_orient"], dtype=torch.float32, device=device)
	pose_body = torch.as_tensor(motion["pose_body"], dtype=torch.float32, device=device)
	pose_hand = None
	pose_jaw = None
	pose_eye = None
	expression = None
	if "pose_hand" in motion:
		pose_hand = torch.as_tensor(motion["pose_hand"], dtype=torch.float32, device=device)
	if "pose_jaw" in motion:
		pose_jaw = torch.as_tensor(motion["pose_jaw"], dtype=torch.float32, device=device)
	if "pose_eye" in motion:
		pose_eye = torch.as_tensor(motion["pose_eye"], dtype=torch.float32, device=device)
	if "expression" in motion:
		expression = torch.as_tensor(motion["expression"], dtype=torch.float32, device=device)

	joint_chunks: list[np.ndarray] = []
	with torch.no_grad():
		chunk_starts = range(0, num_frames, chunk_size)
		if progress_desc is not None:
			chunk_starts = iter_progress(
				chunk_starts,
				total=(num_frames + chunk_size - 1) // chunk_size,
				desc=progress_desc,
				unit="chunk",
			)
		for start in chunk_starts:
			stop = min(start + chunk_size, num_frames)
			batch_size = stop - start
			model_inputs = {
				"betas": betas.expand(batch_size, -1),
				"transl": trans[start:stop],
				"global_orient": root_orient[start:stop],
				"body_pose": pose_body[start:stop],
				"return_verts": False,
			}
			if motion["surface_model_type"] == "smplx":
				expression_dim = int(getattr(body_model, "num_expression_coeffs", 10))
				if expression is not None:
					model_inputs["expression"] = expression[start:stop, :expression_dim]
				else:
					model_inputs["expression"] = torch.zeros(
						(batch_size, expression_dim),
						dtype=torch.float32,
						device=device,
					)
				if pose_hand is not None and pose_hand.shape[1] == 90:
					model_inputs["left_hand_pose"] = pose_hand[start:stop, :45]
					model_inputs["right_hand_pose"] = pose_hand[start:stop, 45:]
				if pose_jaw is not None and pose_jaw.shape[1] == 3:
					model_inputs["jaw_pose"] = pose_jaw[start:stop]
				if pose_eye is not None and pose_eye.shape[1] == 6:
					model_inputs["leye_pose"] = pose_eye[start:stop, :3]
					model_inputs["reye_pose"] = pose_eye[start:stop, 3:]
				# smplx registers default *_pose params with batch size 1; if the motion
				# file omits them, pass explicit zero tensors so every input shares the
				# current batch size (otherwise torch.cat inside SMPL-X fails).
				def _zeros(dim: int) -> torch.Tensor:
					return torch.zeros(
						(batch_size, dim), dtype=torch.float32, device=device
					)

				model_inputs.setdefault("jaw_pose", _zeros(3))
				model_inputs.setdefault("leye_pose", _zeros(3))
				model_inputs.setdefault("reye_pose", _zeros(3))
				left_hand_dim = int(
					getattr(body_model, "num_pca_comps", 0)
				) if getattr(body_model, "use_pca", False) else 45
				model_inputs.setdefault("left_hand_pose", _zeros(left_hand_dim))
				model_inputs.setdefault("right_hand_pose", _zeros(left_hand_dim))
			output = body_model(
				**model_inputs,
			)
			joint_chunks.append(output.joints[:, :24].detach().cpu().numpy())

	joints = np.concatenate(joint_chunks, axis=0)
	if joints.shape[1] < 22:
		raise ValueError(f"SMPL output must expose at least 22 joints, got {joints.shape}")
	return joints


def compute_world_quaternions(motion: dict[str, np.ndarray]) -> np.ndarray:
	num_frames = motion["root_orient"].shape[0]
	local_rotvecs = np.zeros((num_frames, 22, 3), dtype=np.float32)
	local_rotvecs[:, 0, :] = motion["root_orient"]
	local_rotvecs[:, 1:, :] = motion["pose_body"].reshape(num_frames, 21, 3)

	local_rotations = Rotation.from_rotvec(local_rotvecs.reshape(-1, 3)).as_matrix()
	local_rotations = local_rotations.reshape(num_frames, 22, 3, 3)

	world_rotations = np.zeros_like(local_rotations)
	world_rotations[:, 0, :, :] = local_rotations[:, 0, :, :]
	for joint_idx in range(1, 22):
		parent_idx = int(SMPL_PARENTS[joint_idx])
		world_rotations[:, joint_idx, :, :] = np.einsum(
			"nij,njk->nik",
			world_rotations[:, parent_idx, :, :],
			local_rotations[:, joint_idx, :, :],
		)

	quat_xyzw = Rotation.from_matrix(world_rotations.reshape(-1, 3, 3)).as_quat()
	quat_xyzw = quat_xyzw.reshape(num_frames, 22, 4)
	return quat_xyzw[:, :, [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(quaternions: np.ndarray) -> np.ndarray:
	return quaternions[:, :, [1, 2, 3, 0]]


def quat_xyzw_to_wxyz(quaternions: np.ndarray) -> np.ndarray:
	return quaternions[:, :, [3, 0, 1, 2]]


def average_quaternions_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
	# Align hemisphere before averaging to avoid cancellation.
	dot = np.sum(q1 * q2, axis=-1, keepdims=True)
	q2_aligned = np.where(dot < 0.0, -q2, q2)
	q_avg = q1 + q2_aligned
	q_avg /= np.clip(np.linalg.norm(q_avg, axis=-1, keepdims=True), 1e-8, None)
	return q_avg.astype(np.float32)


def apply_axis_map_and_local_euler_offset_wxyz(
	quaternions_wxyz: np.ndarray,
	axis_map: np.ndarray,
	euler_xyz_degrees: np.ndarray,
) -> np.ndarray:
	base_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]
	base_mats = Rotation.from_quat(base_xyzw).as_matrix()
	if axis_map.shape != (3, 3):
		raise ValueError(f"axis_map shape must be (3, 3), got {axis_map.shape}")
	axis_map64 = axis_map.astype(np.float64)
	euler_rad = np.radians(euler_xyz_degrees.astype(np.float64))
	rx, ry, rz = float(euler_rad[0]), float(euler_rad[1]), float(euler_rad[2])
	rot_x = np.array(
		[[1.0, 0.0, 0.0], [0.0, np.cos(rx), -np.sin(rx)], [0.0, np.sin(rx), np.cos(rx)]],
		dtype=np.float64,
	)
	rot_y = np.array(
		[[np.cos(ry), 0.0, np.sin(ry)], [0.0, 1.0, 0.0], [-np.sin(ry), 0.0, np.cos(ry)]],
		dtype=np.float64,
	)
	rot_z = np.array(
		[[np.cos(rz), -np.sin(rz), 0.0], [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]],
		dtype=np.float64,
	)
	offset_mat = rot_x @ rot_y @ rot_z
	# Right multiply in two stages: axis remap first, then xyz local fine-tuning.
	adjusted_mats = np.einsum("nij,jk->nik", base_mats, axis_map64 @ offset_mat)
	adjusted_xyzw = Rotation.from_matrix(adjusted_mats).as_quat()
	adjusted_wxyz = adjusted_xyzw[:, [3, 0, 1, 2]]
	return adjusted_wxyz.astype(np.float32)


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
	norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
	return vectors / np.clip(norms, eps, None)


def quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
	v_from = normalize_vectors(v_from)
	v_to = normalize_vectors(v_to)
	dot = float(np.clip(np.dot(v_from, v_to), -1.0, 1.0))

	if dot > 1.0 - 1e-7:
		return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

	if dot < -1.0 + 1e-7:
		fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
		if abs(v_from[0]) > 0.9:
			fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		axis = normalize_vectors(np.cross(v_from, fallback).reshape(1, 3))[0]
		return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float32)

	cross = np.cross(v_from, v_to)
	quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float32)
	quat /= np.linalg.norm(quat)
	return quat


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	return np.array(
		[
			w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
			w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
			w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
			w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
		],
		dtype=np.float32,
	)


def quat_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
	return np.array(
		[quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
		dtype=np.float32,
	)


def quat_rotate_vector_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
	vector_quat = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float32)
	rotated = quat_multiply_wxyz(
		quat_multiply_wxyz(quaternion, vector_quat),
		quat_conjugate_wxyz(quaternion),
	)
	return rotated[1:]


def align_link_quaternions(
	joint_positions: np.ndarray,
	joint_quaternions: np.ndarray,
) -> np.ndarray:
	original = joint_quaternions.copy()
	aligned = joint_quaternions.copy()
	identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
	body_names = list(SKELETON_TO_SMPL.keys())
	body_slots = {name: idx for idx, name in enumerate(body_names)}

	for body_name, child_name in BODY_CHILDREN.items():
		body_slot = body_slots[body_name]
		child_slot = body_slots[child_name]
		local_dir = BODY_LOCAL_DIRECTIONS[body_name]

		for frame_idx in range(joint_positions.shape[0]):
			bone_vec = joint_positions[frame_idx, child_slot] - joint_positions[frame_idx, body_slot]
			if np.linalg.norm(bone_vec) < 1e-8:
				aligned[frame_idx, body_slot] = identity
				continue
			original_quat = original[frame_idx, body_slot]
			original_axis_world = quat_rotate_vector_wxyz(original_quat, local_dir)
			correction = quat_from_two_vectors(original_axis_world, bone_vec)
			aligned[frame_idx, body_slot] = quat_multiply_wxyz(correction, original_quat)

	return aligned


def select_frame_slice(num_frames: int, start_frame: int, end_frame: int, stride: int) -> np.ndarray:
	start = max(0, start_frame)
	stop = num_frames if end_frame < 0 else min(num_frames, end_frame + 1)
	if start >= stop:
		raise ValueError(f"Empty frame range: start={start_frame}, end={end_frame}, total={num_frames}")
	if stride <= 0:
		raise ValueError(f"Stride must be positive, got {stride}")
	return np.arange(start, stop, stride, dtype=np.int32)


def symmetrize_hip_joints(
	positions: np.ndarray,
	root_quaternions_wxyz: np.ndarray,
	body_slots: dict[str, int],
	left_body: str = "left_up_leg",
	right_body: str = "right_up_leg",
	lateral_axis: int = 0,
) -> np.ndarray:
	"""Mirror-symmetrize the left/right hip joints in the root-local frame.

	The SMPL/SMPLX joint regressor is not perfectly left/right symmetric, so the
	rest-pose hips differ by ~1 cm. Instead of clamping a fixed world axis (which
	collapses the hip spacing once the body is rotated), this transforms both hip
	offsets into the pelvis-local frame, enforces a perfect left/right mirror
	(the lateral axis is flipped, the other two components are averaged), then
	maps the result back to world space. This keeps the lateral hip width intact
	and stays correct for any body orientation.
	"""
	if left_body not in body_slots or right_body not in body_slots:
		raise ValueError(f"Missing hip body in replay buffers: {left_body} / {right_body}")
	if lateral_axis not in (0, 1, 2):
		raise ValueError(f"lateral_axis must be 0, 1 or 2, got {lateral_axis}")
	if root_quaternions_wxyz.shape != (positions.shape[0], 4):
		raise ValueError(
			"root_quaternions_wxyz shape must be [T, 4], got "
			f"{root_quaternions_wxyz.shape} for T={positions.shape[0]}"
		)

	left_idx = body_slots[left_body]
	right_idx = body_slots[right_body]

	# Pelvis center and its world rotation (root is SMPL joint 0 / pelvis).
	pelvis_center = 0.5 * (positions[:, left_idx, :] + positions[:, right_idx, :])
	root_rot = Rotation.from_quat(root_quaternions_wxyz[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)

	# Express hip offsets in the pelvis-local frame: local = R^T @ (world - center).
	left_world_offset = positions[:, left_idx, :] - pelvis_center
	right_world_offset = positions[:, right_idx, :] - pelvis_center
	left_local = np.einsum("nji,nj->ni", root_rot, left_world_offset)
	right_local = np.einsum("nji,nj->ni", root_rot, right_world_offset)

	# Enforce a perfect mirror about the sagittal plane in local coordinates.
	mirrored_right = right_local.copy()
	mirrored_right[:, lateral_axis] *= -1.0
	symmetric_left = 0.5 * (left_local + mirrored_right)
	symmetric_right = symmetric_left.copy()
	symmetric_right[:, lateral_axis] *= -1.0

	# Map the symmetric offsets back to world space.
	positions[:, left_idx, :] = pelvis_center + np.einsum("nij,nj->ni", root_rot, symmetric_left)
	positions[:, right_idx, :] = pelvis_center + np.einsum("nij,nj->ni", root_rot, symmetric_right)
	return positions


def build_replay_buffers(
	motion_file: Path,
	smpl_model_path: Path,
	gender_override: str,
	device: str,
	chunk_size: int,
	translation_offset: np.ndarray,
	model_type_override: str = "auto",
	translation_scale: float = 1.0,
	up_axis: str = "auto",
	target_fps: float = 0.0,
	max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, str, dict[str, object]]:
	motion, inferred_gender, fps = load_motion_arrays(
		motion_file,
		model_type_override=model_type_override,
		translation_scale=translation_scale,
		up_axis=up_axis,
		target_fps=target_fps,
		max_frames=max_frames,
	)
	gender = normalize_gender(inferred_gender, gender_override)
	num_betas = int(min(10, max(1, motion["betas"].shape[0])))

	body_model = build_smpl_model(
		smpl_model_path,
		motion["surface_model_type"],
		gender,
		num_betas,
		device,
	)
	joint_positions = compute_joint_positions(
		motion,
		body_model,
		device,
		chunk_size,
		progress_desc="SMPL forward",
	)
	joint_quaternions = compute_world_quaternions(motion)

	base_body_names = list(SKELETON_TO_SMPL.keys())
	base_body_count = len(base_body_names)
	body_slots = {name: idx for idx, name in enumerate(REPLAY_BODY_NAMES)}
	positions = np.zeros((joint_positions.shape[0], len(REPLAY_BODY_NAMES), 3), dtype=np.float32)
	quaternions = np.zeros((joint_quaternions.shape[0], len(REPLAY_BODY_NAMES), 4), dtype=np.float32)
	identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

	for body_idx, (_body_name, smpl_idx) in enumerate(SKELETON_TO_SMPL.items()):
		positions[:, body_idx, :] = joint_positions[:, smpl_idx, :] + translation_offset
		quaternions[:, body_idx, :] = joint_quaternions[:, smpl_idx, :]

	# SMPL/SMPLX joints are not perfectly left/right symmetric; level the hips.
	positions = symmetrize_hip_joints(
		positions,
		quaternions[:, body_slots["hips"], :],
		body_slots,
	)

	for body_name, (left_name, right_name) in DERIVED_BODY_CENTERS.items():
		body_idx = body_slots[body_name]
		left_idx = body_slots[left_name]
		right_idx = body_slots[right_name]
		positions[:, body_idx, :] = 0.5 * (positions[:, left_idx, :] + positions[:, right_idx, :])
		quaternions[:, body_idx, :] = average_quaternions_wxyz(
			quaternions[:, left_idx, :],
			quaternions[:, right_idx, :],
		)

	quaternions[:, :base_body_count, :] = align_link_quaternions(
		positions[:, :base_body_count, :],
		quaternions[:, :base_body_count, :],
	)

	metadata = {
		"source_motion_file": str(motion.get("source_motion_file", motion_file)),
		"source_model_type": str(motion["surface_model_type"]),
		"source_fps": float(motion.get("source_fps", fps)),
		"target_fps": float(fps),
		"source_coordinate_system": str(motion.get("source_coordinate_system", "")),
	}
	return positions, quaternions, fps, gender, metadata


def print_summary(
	motion_file: Path,
	positions: np.ndarray,
	fps: float,
	gender: str,
	selected_frames: np.ndarray,
) -> None:
	duration = positions.shape[0] / fps if fps > 0 else 0.0
	print(f"motion_file: {motion_file}")
	print(f"gender: {gender}")
	print(f"frames_total: {positions.shape[0]}")
	print(f"frames_selected: {selected_frames.shape[0]}")
	print(f"fps: {fps:.3f}")
	print(f"duration_sec: {duration:.3f}")


def save_keypoints_pkl(
	output_path: Path,
	keypoint_names: list[str],
	positions: np.ndarray,
	quaternions: np.ndarray,
	fps: float | None = None,
	contact_names: tuple[str, ...] | None = None,
	contact_positions: np.ndarray | None = None,
	contact_speeds: np.ndarray | None = None,
	contact_states: np.ndarray | None = None,
	contact_vel_window: int | None = None,
	contact_vel_threshold: float | None = None,
	contact_height_threshold: float | None = None,
	metadata: dict[str, object] | None = None,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"keypoint_names": keypoint_names,
		"positions": positions.astype(np.float32),
		"quaternions": quaternions.astype(np.float32),
	}
	if fps is not None:
		payload["fps"] = float(fps)
	if contact_names is not None:
		payload["contact_names"] = list(contact_names)
	if contact_positions is not None:
		payload["contact_positions"] = contact_positions.astype(np.float32)
	if contact_speeds is not None:
		payload["contact_speeds"] = contact_speeds.astype(np.float32)
	if contact_states is not None:
		payload["contact_states"] = contact_states.astype(np.bool_)
	if contact_vel_window is not None:
		payload["contact_vel_window"] = int(contact_vel_window)
	if contact_vel_threshold is not None:
		payload["contact_vel_threshold"] = float(contact_vel_threshold)
	if contact_height_threshold is not None:
		payload["contact_height_threshold"] = float(contact_height_threshold)
	if metadata:
		for key, value in metadata.items():
			if key in payload:
				raise ValueError(f"keypoint 元数据不得覆盖核心字段: field={key}, path={output_path}")
			payload[key] = value
	with output_path.open("wb") as f:
		pickle.dump(payload, f)
	
	print(f"Saved keypoints to: {output_path}")


def advance_frame_cursor(cursor: int, delta: int, num_frames: int, loop: bool) -> int:
	if num_frames <= 0:
		raise ValueError("num_frames must be positive")

	updated = cursor + delta
	if loop:
		return updated % num_frames
	return int(np.clip(updated, 0, num_frames - 1))


def format_progress_line(
	cursor: int,
	num_frames: int,
	frame_ids: np.ndarray,
	paused: bool,
	bar_width: int = 30,
) -> str:
	progress = (cursor + 1) / max(num_frames, 1)
	filled = min(bar_width, max(0, int(round(progress * bar_width))))
	bar = "#" * filled + "-" * (bar_width - filled)
	status = "PAUSED" if paused else "PLAY"
	source_frame = int(frame_ids[cursor])
	return (
		f"\r[{bar}] {cursor + 1:4d}/{num_frames:4d} "
		f"src={source_frame:4d} {status:<6}"
	)


def play_clip(
	model: mujoco.MjModel,
	data: mujoco.MjData,
	positions: np.ndarray,
	quaternions: np.ndarray,
	retarget_keypoints: np.ndarray | None,
	retarget_keypoint_quaternions: np.ndarray | None,
	contact_positions: np.ndarray | None,
	contact_states: np.ndarray | None,
	body_names: list[str],
	fps: float,
	frame_ids: np.ndarray,
	loop: bool,
) -> None:
	frame_time = 1.0 / fps
	qpos_addresses = []
	for body_name in body_names:
		joint_name = f"{body_name}_joint"
		joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
		if joint_id < 0:
			raise ValueError(f"Missing joint in MJCF: {joint_name}")
		qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

	num_frames = int(frame_ids.shape[0])
	state = {
		"paused": False,
		"cursor": 0,
		"step_delta": 0,
		"resync_clock": False,
	}
	state_lock = threading.Lock()

	def key_callback(keycode: int) -> None:
		with state_lock:
			if keycode == glfw.KEY_SPACE:
				state["paused"] = not state["paused"]
				state["resync_clock"] = True
			elif keycode == glfw.KEY_COMMA:
				state["paused"] = True
				state["step_delta"] -= 1
				state["resync_clock"] = True
			elif keycode == glfw.KEY_PERIOD:
				state["paused"] = True
				state["step_delta"] += 1
				state["resync_clock"] = True
			elif keycode == glfw.KEY_LEFT_BRACKET:
				state["paused"] = True
				state["step_delta"] -= 10
				state["resync_clock"] = True
			elif keycode == glfw.KEY_RIGHT_BRACKET:
				state["paused"] = True
				state["step_delta"] += 10
				state["resync_clock"] = True
			elif keycode == glfw.KEY_R:
				state["cursor"] = 0
				state["step_delta"] = 0
				state["resync_clock"] = True

	def render_cursor(cursor: int, viewer=None) -> None:
		frame_idx = int(frame_ids[cursor])
		for body_slot, qpos_addr in enumerate(qpos_addresses):
			data.qpos[qpos_addr : qpos_addr + 3] = positions[frame_idx, body_slot, :]
			data.qpos[qpos_addr + 3 : qpos_addr + 7] = quaternions[frame_idx, body_slot, :]
		mujoco.mj_forward(model, data)
		if viewer is not None and retarget_keypoints is not None:
			update_viewer_keypoints(
				viewer,
				retarget_keypoints[frame_idx],
				None if retarget_keypoint_quaternions is None else retarget_keypoint_quaternions[frame_idx],
				contact_keypoints=None if contact_positions is None else contact_positions[frame_idx],
				contact_states=None if contact_states is None else contact_states[frame_idx],
			)

	print("Controls: Space play/pause, ',' back 1, '.' forward 1, '[' back 10, ']' forward 10, 'R' reset")
	with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
		last_advance_time = time.perf_counter()
		last_progress_line = ""
		while viewer.is_running():
			frame_start = time.perf_counter()
			with state_lock:
				paused = bool(state["paused"])
				cursor = int(state["cursor"])
				step_delta = int(state["step_delta"])
				resync_clock = bool(state["resync_clock"])
				state["step_delta"] = 0
				state["resync_clock"] = False

			if resync_clock:
				last_advance_time = time.perf_counter()

			if step_delta != 0:
				cursor = advance_frame_cursor(cursor, step_delta, num_frames, loop)
				paused = True
				with state_lock:
					state["paused"] = True
					state["cursor"] = cursor
				last_advance_time = time.perf_counter()
			elif not paused:
				now = time.perf_counter()
				if now - last_advance_time >= frame_time:
					if cursor == num_frames - 1 and not loop:
						render_cursor(cursor, viewer)
						viewer.sync()
						return
					cursor = advance_frame_cursor(cursor, 1, num_frames, loop)
					with state_lock:
						state["cursor"] = cursor
					last_advance_time = now

			render_cursor(cursor, viewer)
			progress_line = format_progress_line(cursor, num_frames, frame_ids, paused)
			if progress_line != last_progress_line:
				sys.stdout.write(progress_line)
				sys.stdout.flush()
				last_progress_line = progress_line
			viewer.sync()

			elapsed = time.perf_counter() - frame_start
			if paused or step_delta != 0:
				time.sleep(0.01)
			else:
				remaining = frame_time - elapsed
				if remaining > 0.0:
					time.sleep(remaining)
		sys.stdout.write("\n")
		sys.stdout.flush()


def main() -> None:
	args = parse_args()
	robot_config_path = args.robot_config.expanduser().resolve()
	with robot_config_path.open("r", encoding="utf-8") as file:
		robot_config = yaml.safe_load(file) or {}
	robot_mjcf_path = load_path_config(
		robot_config_path,
		"robot_xml_path"
	).expanduser().resolve()
	skeleton_config_path = args.skeleton_config.expanduser().resolve()
	skeleton_mjcf_path = args.mjcf.expanduser().resolve()
	robot_link_lengths = compute_robot_link_lengths(
		config_path=robot_config_path,
		robot_mjcf_path=robot_mjcf_path,
	)

	positions, quaternions, inferred_fps, gender, motion_metadata = build_replay_buffers(
		motion_file=args.motion_file.expanduser().resolve(),
		smpl_model_path=args.smpl_model_path.expanduser().resolve(),
		gender_override=args.gender,
		device=args.device,
		chunk_size=args.chunk_size,
		translation_offset=np.asarray(args.translation_offset, dtype=np.float32),
		model_type_override=args.model_type,
		translation_scale=args.translation_scale,
		up_axis=args.up_axis,
		target_fps=args.target_fps,
		max_frames=args.max_frames,
	)
	contact_links = load_yaml_body_list_config(skeleton_config_path, "contact_links")
	ground_reference_contacts = tuple(
		str(name) for name in robot_config.get("ground_reference_contacts", contact_links)
	)
	if len(set(ground_reference_contacts)) != len(ground_reference_contacts):
		raise ValueError(
			f"ground_reference_contacts 含重复项: path={robot_config_path}, actual={ground_reference_contacts}"
		)
	contact_index = {name: index for index, name in enumerate(contact_links)}
	missing_ground_contacts = [name for name in ground_reference_contacts if name not in contact_index]
	if missing_ground_contacts:
		raise ValueError(
			f"ground reference 不在源 contact_names 中: path={robot_config_path}, "
			f"missing={missing_ground_contacts}, actual={list(contact_links)}"
		)
	window_seconds = robot_config.get("contact_vel_calculate_window_seconds")
	if window_seconds is None:
		contact_vel_window = load_scalar_int_config(
			skeleton_config_path,
			"contact_vel_calculate_window",
			default=6,
		)
	else:
		window_seconds = float(window_seconds)
		if not np.isfinite(window_seconds) or window_seconds <= 0.0:
			raise ValueError(
				f"接触速度时间窗口必须为正有限值: path={robot_config_path}, "
				f"field=contact_vel_calculate_window_seconds, actual={window_seconds}"
			)
		contact_vel_window = max(1, round(window_seconds * inferred_fps))
	contact_vel_threshold = load_scalar_float_config(
		skeleton_config_path,
		"contact_vel_threshold",
		default=0.5,
	)
	contact_height_threshold = load_scalar_float_config(
		skeleton_config_path,
		"contact_height_threshold",
		default=0.05,
	)
	contact_height_lpf_alpha = load_scalar_float_config(
		skeleton_config_path,
		"contact_height_lpf_alpha",
		default=0.2,
	)
	relative_sequence_floor_enabled = bool(
		robot_config.get("contact_height_relative_to_sequence_floor", False)
	)
	dynamic_height_offset_enabled = bool(
		robot_config.get("contact_height_dynamic_offset_enabled", True)
	)
	contact_height_floor_method = str(
		robot_config.get("contact_height_floor_method", "percentile")
	).strip()
	height_floor_percentile: float | None = None
	height_floor_estimator: dict[str, object] | None = None
	if relative_sequence_floor_enabled:
		if contact_height_floor_method == "percentile":
			height_floor_percentile = float(
				robot_config["contact_height_floor_percentile"]
			)
		elif contact_height_floor_method == "stable_support_dense_median":
			height_floor_estimator = {
				"speed_threshold_mps": float(
					robot_config["contact_height_floor_fit_speed_threshold_mps"]
				),
				"inlier_tolerance_m": float(
					robot_config["contact_height_floor_fit_inlier_tolerance_m"]
				),
				"minimum_samples": int(
					robot_config["contact_height_floor_fit_min_samples"]
				),
			}
		else:
			raise ValueError(
				"不支持的接触地板估计方法: "
				f"path={robot_config_path}, method={contact_height_floor_method}, "
				"expected=percentile|stable_support_dense_median"
			)
	contact_positions, contact_speeds, contact_states = compute_contact_sequence(
		positions=positions,
		quaternions=quaternions,
		contact_links=contact_links,
		skeleton_mjcf_path=skeleton_mjcf_path,
		fps=inferred_fps,
		vel_window=contact_vel_window,
		vel_threshold=contact_vel_threshold,
		height_threshold=contact_height_threshold,
		height_reference_contact_names=ground_reference_contacts,
		height_floor_percentile=height_floor_percentile,
		height_floor_estimator=height_floor_estimator,
	)
	ground_indices = np.asarray(
		[contact_index[name] for name in ground_reference_contacts], dtype=np.int32
	)
	source_sequence_floor_height = 0.0
	source_floor_fit_report: dict[str, object] = {"method": "disabled"}
	if relative_sequence_floor_enabled:
		if height_floor_estimator is not None:
			source_sequence_floor_height, source_floor_fit_report = (
				fit_stable_support_floor_height(
					contact_positions=contact_positions,
					contact_speeds=contact_speeds,
					reference_indices=ground_indices,
					speed_threshold_mps=float(
						height_floor_estimator["speed_threshold_mps"]
					),
					inlier_tolerance_m=float(
						height_floor_estimator["inlier_tolerance_m"]
					),
					minimum_samples=int(
						height_floor_estimator["minimum_samples"]
					),
				)
			)
		else:
			source_sequence_floor_height = compute_sequence_contact_floor_height(
				contact_positions=contact_positions,
				reference_indices=ground_indices,
				percentile=float(height_floor_percentile),
			)
			source_floor_fit_report = {
				"method": "percentile",
				"floor_height_m": float(source_sequence_floor_height),
				"percentile": float(height_floor_percentile),
			}
	raw_contact_states = contact_states.copy()
	contact_hysteresis_config = robot_config.get("contact_hysteresis", {}) or {}
	if not isinstance(contact_hysteresis_config, dict):
		raise TypeError(
			"contact_hysteresis 必须是映射: "
			f"path={robot_config_path}, actual={type(contact_hysteresis_config).__name__}"
		)
	contact_hysteresis_report: dict[str, object] = {
		"enabled": bool(contact_hysteresis_config.get("enabled", False))
	}
	if contact_hysteresis_report["enabled"]:
		relative_contact_heights = contact_positions[:, :, 2].astype(np.float64)
		if relative_sequence_floor_enabled:
			hysteresis_floor = source_sequence_floor_height
			relative_contact_heights = relative_contact_heights - hysteresis_floor
		else:
			hysteresis_floor = 0.0

		def seconds_to_frames(field_name: str, default: float) -> int:
			seconds = float(contact_hysteresis_config.get(field_name, default))
			if not np.isfinite(seconds) or seconds <= 0.0:
				raise ValueError(
					f"contact_hysteresis.{field_name} 必须为正有限秒数: {seconds}"
				)
			return max(1, int(round(seconds * inferred_fps)))

		enter_velocity = float(
			contact_hysteresis_config.get("enter_speed_mps", 0.35)
		)
		exit_velocity = float(
			contact_hysteresis_config.get("exit_speed_mps", 0.60)
		)
		enter_height = float(
			contact_hysteresis_config.get("enter_relative_height_m", 0.025)
		)
		exit_height = float(
			contact_hysteresis_config.get("exit_relative_height_m", 0.05)
		)
		enter_frames = seconds_to_frames("enter_confirmation_seconds", 0.06)
		exit_frames = seconds_to_frames("exit_confirmation_seconds", 0.08)
		minimum_contact_frames = seconds_to_frames("minimum_contact_seconds", 0.12)
		minimum_swing_frames = seconds_to_frames("minimum_swing_seconds", 0.08)
		contact_states = stabilize_contact_states(
			contact_speeds=contact_speeds,
			relative_heights=relative_contact_heights,
			enter_velocity=enter_velocity,
			exit_velocity=exit_velocity,
			enter_height=enter_height,
			exit_height=exit_height,
			enter_confirmation_frames=enter_frames,
			exit_confirmation_frames=exit_frames,
			minimum_contact_frames=minimum_contact_frames,
			minimum_swing_frames=minimum_swing_frames,
		)
		contact_hysteresis_report.update(
			{
				"enter_speed_mps": enter_velocity,
				"exit_speed_mps": exit_velocity,
				"enter_relative_height_m": enter_height,
				"exit_relative_height_m": exit_height,
				"enter_confirmation_frames": enter_frames,
				"exit_confirmation_frames": exit_frames,
				"minimum_contact_frames": minimum_contact_frames,
				"minimum_swing_frames": minimum_swing_frames,
				"sequence_floor_m": float(hysteresis_floor),
			}
		)
	contact_hysteresis_report.update(
		{
			"raw_transition_count": int(
				np.count_nonzero(raw_contact_states[1:] != raw_contact_states[:-1])
			),
			"filtered_transition_count": int(
				np.count_nonzero(contact_states[1:] != contact_states[:-1])
			),
			"raw_active_frames_per_contact": raw_contact_states.sum(axis=0)
			.astype(int)
			.tolist(),
			"filtered_active_frames_per_contact": contact_states.sum(axis=0)
			.astype(int)
			.tolist(),
		}
	)
	(
		retarget_keypoints,
		retarget_keypoint_quaternions,
		skeleton_link_vectors,
		link_scales,
		link_scale_is_static,
	) = build_retarget_keypoints(
		skeleton_positions=positions,
		skeleton_quaternions=quaternions,
		robot_link_lengths=robot_link_lengths,
		robot_config_path=robot_config_path,
		skeleton_config_path=skeleton_config_path,
	)
	robot_links = load_robot_links_config(robot_config_path)
	knee_angle_offset_degrees = load_scalar_float_config(
		robot_config_path,
		"knee_angle_offset_degrees",
		default=15.0,
	)
	(
		retarget_keypoints,
		retarget_keypoint_quaternions,
		extra_keypoint_names,
	) = append_robot_foot_keypoints(
		keypoints=retarget_keypoints,
		keypoint_quaternions=retarget_keypoint_quaternions,
		robot_links=robot_links,
		robot_config_path=robot_config_path,
		robot_mjcf_path=robot_mjcf_path,
	)
	leg_displacement_scale, robot_leg_length, skeleton_leg_length = compute_leg_displacement_scale(
		robot_link_lengths=robot_link_lengths,
		skeleton_link_vectors=skeleton_link_vectors,
		knee_angle_offset_degrees=knee_angle_offset_degrees,
	)
	retarget_keypoints = scale_keypoint_frame_displacements(
		keypoints=retarget_keypoints,
		displacement_scale=leg_displacement_scale,
		root_keypoint_idx=0,
	)
	keypoint_names = ["hips_mean", *list(robot_links.keys()), *extra_keypoint_names]
	retarget_sequence_floor_height = 0.0
	retarget_floor_fit_report: dict[str, object] = {"method": "disabled"}
	if relative_sequence_floor_enabled:
		retarget_ground_indices = resolve_contact_height_reference_indices(
			keypoint_names,
			ground_reference_contacts,
		)
		if np.any(retarget_ground_indices < 0):
			raise ValueError(
				"ground reference 无法映射到缩放后的机器人关键点: "
				f"contacts={ground_reference_contacts}, indices={retarget_ground_indices.tolist()}"
			)
		if height_floor_estimator is not None:
			retarget_reference_positions = retarget_keypoints[
				:, retarget_ground_indices, :
			]
			retarget_sequence_floor_height, retarget_floor_fit_report = (
				fit_stable_support_floor_height(
					contact_positions=retarget_reference_positions,
					contact_speeds=contact_speeds[:, ground_indices],
					reference_indices=np.arange(
						len(ground_reference_contacts), dtype=np.int32
					),
					speed_threshold_mps=float(
						height_floor_estimator["speed_threshold_mps"]
					),
					inlier_tolerance_m=float(
						height_floor_estimator["inlier_tolerance_m"]
					),
					minimum_samples=int(
						height_floor_estimator["minimum_samples"]
					),
				)
			)
		else:
			retarget_sequence_floor_height = compute_sequence_contact_floor_height(
				contact_positions=retarget_keypoints,
				reference_indices=retarget_ground_indices,
				percentile=float(height_floor_percentile),
			)
			retarget_floor_fit_report = {
				"method": "percentile",
				"floor_height_m": float(retarget_sequence_floor_height),
				"percentile": float(height_floor_percentile),
			}
	retarget_keypoints, contact_height_offsets = offset_keypoints_by_contact_height(
		keypoints=retarget_keypoints,
		keypoint_names=keypoint_names,
		contact_names=ground_reference_contacts,
		contact_positions=contact_positions[:, ground_indices, :],
		contact_states=contact_states[:, ground_indices],
		height_lpf_alpha=contact_height_lpf_alpha,
		prevent_penetration=bool(
			robot_config.get("contact_height_prevent_penetration", False)
		),
		initial_height=retarget_sequence_floor_height,
		dynamic_offset_enabled=dynamic_height_offset_enabled,
	)
	keypoints_stem = args.keypoints_name or args.motion_file.expanduser().resolve().stem
	keypoint_output_path = args.output_dir.expanduser().resolve() / "keypoints" / robot_config_path.stem / (
		f"{keypoints_stem}_keypoints.pkl"
	)
	save_keypoints_pkl(
		output_path=keypoint_output_path,
		keypoint_names=keypoint_names,
		positions=retarget_keypoints,
		quaternions=retarget_keypoint_quaternions,
		fps=inferred_fps,
		contact_names=contact_links,
		contact_positions=contact_positions,
		contact_speeds=contact_speeds,
		contact_states=contact_states,
		contact_vel_window=contact_vel_window,
		contact_vel_threshold=contact_vel_threshold,
		contact_height_threshold=contact_height_threshold,
		metadata={
			**motion_metadata,
			"robot_config_name": robot_config_path.stem,
			"robot_link_lengths": robot_link_lengths,
			"leg_displacement_scale": leg_displacement_scale,
			"ground_reference_contacts": list(ground_reference_contacts),
			"contact_height_dynamic_offset_enabled": dynamic_height_offset_enabled,
			"contact_height_offset_mode": (
				"dynamic_contact"
				if dynamic_height_offset_enabled
				else (
					"static_sequence_floor"
					if relative_sequence_floor_enabled
					else "disabled"
				)
			),
			"contact_height_prevent_penetration": bool(
				robot_config.get("contact_height_prevent_penetration", False)
			),
			"contact_height_relative_to_sequence_floor": relative_sequence_floor_enabled,
			"contact_height_floor_method": (
				contact_height_floor_method
				if relative_sequence_floor_enabled
				else "disabled"
			),
			"contact_height_floor_percentile": robot_config.get(
				"contact_height_floor_percentile"
			),
			"contact_height_source_sequence_floor": float(source_sequence_floor_height),
			"contact_height_retarget_sequence_floor": float(retarget_sequence_floor_height),
			"contact_height_source_floor_fit": source_floor_fit_report,
			"contact_height_retarget_floor_fit": retarget_floor_fit_report,
			"contact_height_offset_min": float(np.min(contact_height_offsets)),
			"contact_height_offset_median": float(np.median(contact_height_offsets)),
			"contact_height_offset_max": float(np.max(contact_height_offsets)),
			"contact_hysteresis": contact_hysteresis_report,
		},
	)

	fps = float(args.fps) if args.fps > 0.0 else inferred_fps
	if fps <= 0.0:
		raise ValueError(f"Invalid playback fps: {fps}")

	frame_ids = select_frame_slice(positions.shape[0], args.start_frame, args.end_frame, args.stride)
	if args.print_summary:
		print_summary(args.motion_file, positions, fps, gender, frame_ids)
		print(f"contact_names: {list(contact_links)}")
		print(f"contact_states_shape: {contact_states.shape}")
		print(f"contact_height_lpf_alpha: {contact_height_lpf_alpha}")
		print(f"knee_angle_offset_degrees: {knee_angle_offset_degrees}")
		print(f"robot_leg_length: {robot_leg_length}")
		print(f"skeleton_leg_length: {skeleton_leg_length}")
		print(f"leg_displacement_scale: {leg_displacement_scale}")
		print(f"keypoints_pkl: {keypoint_output_path}")
		print(f"robot_mjcf_path: {robot_mjcf_path}")
		print(f"robot_link_lengths: {robot_link_lengths}")
		print(f"retarget_keypoints_shape: {retarget_keypoints.shape}")
		print(f"retarget_keypoint_quaternions_shape: {retarget_keypoint_quaternions.shape}")
		print(f"contact_height_offsets_shape: {contact_height_offsets.shape}")
		print(f"skeleton_link_vectors: {skeleton_link_vectors}")
		print(f"link_scales: {link_scales}")
		print(f"link_scale_is_static: {link_scale_is_static}")

	if args.no_viewer:
		return

	mjcf_path = skeleton_mjcf_path
	if not mjcf_path.exists():
		raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")

	model = mujoco.MjModel.from_xml_path(str(mjcf_path))
	data = mujoco.MjData(model)
	play_clip(
		model=model,
		data=data,
		positions=positions,
		quaternions=quaternions,
		retarget_keypoints=retarget_keypoints,
		retarget_keypoint_quaternions=retarget_keypoint_quaternions,
		contact_positions=contact_positions,
		contact_states=contact_states,
		body_names=REPLAY_BODY_NAMES,
		fps=fps,
		frame_ids=frame_ids,
		loop=args.loop,
	)


if __name__ == "__main__":
	main()
