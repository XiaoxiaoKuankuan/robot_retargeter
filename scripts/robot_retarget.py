"""Retarget keypoint motions to a robot and export MuJoCo qpos CSV.

This script loads robot and keypoint settings from a robot YAML config,
runs IK-based retargeting frame by frame, and saves the resulting motion
as a CSV under output_data/robot_motion.

Usage:
    # Run with a specific robot config from terminal
    python scripts/robot_retarget.py --config config/robot/h2.yaml

    # Override render_debug from terminal (highest priority)
    python scripts/robot_retarget.py --config config/robot/agibot_x2.yaml --render-debug

    # Override keypoints_path from terminal by motion stem only
    python scripts/robot_retarget.py --config config/robot/h2.yaml --keypoints-name body_check_001__A548_M_from_g1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mink
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import pickle
import time
import csv
import copy
import os
import glfw
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global pause state and keyboard callback
_PAUSED = False


@dataclass(frozen=True)
class ContactMapping:
    """Validated mapping from a source contact flag to a robot body task."""

    source_contact_name: str
    frame_name: str
    source_keypoint_name: str
    frame_type: str
    position_cost: float


class HeelToeHeightDifferenceTask(mink.Task):
    """软约束同一只脚的脚跟与脚尖世界高度相等。

    任务误差只有一个标量 ``heel_z - toe_z``，雅可比是两个足点世界位置雅可比
    的 Z 行之差。freejoint 的统一 Root-Z 平移在相减后严格抵消，所以该任务只会
    调整腿部/脚踝相对构型，不会直接把整台机器人上下搬动。任务权重由外部根据
    heel 与 toe 同时接触的状态平滑渐入渐出；零权重时误差返回零，避免旧版
    ``legacy_raw`` 收敛判据被一个未激活的软任务影响。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        heel_body_name: str,
        toe_body_name: str,
        cost: float = 0.0,
        lm_damping: float = 1.0,
    ):
        self.model = model
        self.heel_body_name = str(heel_body_name)
        self.toe_body_name = str(toe_body_name)
        self.heel_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.heel_body_name
        )
        self.toe_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.toe_body_name
        )
        if self.heel_body_id < 0 or self.toe_body_id < 0:
            raise ValueError(
                "heel/toe 高度差任务找不到机器人 body: "
                f"heel={self.heel_body_name}, toe={self.toe_body_name}"
            )
        super().__init__(
            cost=np.asarray([0.0], dtype=np.float64),
            gain=1.0,
            lm_damping=float(lm_damping),
        )
        self.set_height_cost(cost)

    def set_height_cost(self, cost: float) -> None:
        cost = float(cost)
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError(f"heel/toe 高度差 cost 必须非负且有限: {cost}")
        self.cost = np.asarray([cost], dtype=np.float64)

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        if self.cost[0] <= 0.0:
            return np.zeros(1, dtype=np.float64)
        heel_z = float(configuration.data.xpos[self.heel_body_id, 2])
        toe_z = float(configuration.data.xpos[self.toe_body_id, 2])
        return np.asarray([heel_z - toe_z], dtype=np.float64)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        heel_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        toe_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        rotational_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(
            self.model,
            configuration.data,
            heel_jacobian,
            rotational_jacobian,
            self.heel_body_id,
        )
        mujoco.mj_jacBody(
            self.model,
            configuration.data,
            toe_jacobian,
            rotational_jacobian,
            self.toe_body_id,
        )
        return (heel_jacobian[2] - toe_jacobian[2])[None, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget keypoint motions to a robot from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("config", "robot", "h2.yaml"),
        help="Robot YAML config path (default: config/robot/h2.yaml)",
    )
    parser.add_argument(
        "--keypoints-name",
        type=str,
        default=None,
        help=(
            "Override keypoints_path by motion stem only. "
            "Example: --keypoints-name body_check_001__A548_M_from_g1 -> "
            "output_data/keypoints/<config_stem>/<name>_keypoints.pkl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_data",
        help="Output root containing keypoints/ and robot_motion/ (default: output_data).",
    )
    render_debug_group = parser.add_mutually_exclusive_group()
    render_debug_group.add_argument(
        "--render-debug",
        dest="render_debug",
        action="store_true",
        help="Force enable MuJoCo debug viewer (overrides YAML render_debug).",
    )
    render_debug_group.add_argument(
        "--no-render-debug",
        dest="render_debug",
        action="store_false",
        help="Force disable MuJoCo debug viewer (overrides YAML render_debug).",
    )
    parser.set_defaults(render_debug=None)
    return parser.parse_args()


def key_callback(keycode: int) -> None:
    """Toggle pause/play with the space key."""
    global _PAUSED
    if keycode == glfw.KEY_SPACE:
        _PAUSED = not _PAUSED


class RobotRetarget:

    LEGACY_CONTACT_CONFIG_TO_NAMES = {
        "foot_end_link": ("left_foot_end", "right_foot_end"),
        "toe_link": ("left_toe", "right_toe"),
        "hand_link": ("left_hand", "right_hand"),
    }

    def __init__(
        self,
        model_path: str,
        keypoint_path: str,
        ik_match_table: dict,
        solver: str = "daqp",
        verbose: bool = False,
        damping: float = 1.0,
        render_debug: bool = False,
        joints_limit_offset_degrees: dict | None = None,
        contact_body_names: list | tuple | dict | None = None,
        contact_position_cost: float = 10.0,
        contact_map: dict | None = None,
        contact_task_mode: str = "legacy_hold",
        contact_position_aggregation: str = "mean",
        contact_transition_window_seconds: float = 0.0,
        contact_weight_ramp_seconds: float = 0.0,
        paired_flat_orientation_cost: float = 0.0,
        heel_toe_height_difference_cost: float = 0.0,
        heel_toe_height_difference_ramp_seconds: float = 0.0,
        initial_root_pose: dict | None = None,
        initial_joint_positions: dict[str, float] | None = None,
        max_ik_iterations: int = 50,
        initial_settle_iterations: int = 0,
        ik_error_improvement_threshold: float = 0.001,
        ik_error_metric: str = "legacy_raw",
        robot_name: str = "robot",
        config_path: str | None = None,
        isaac_joint_names: list[str] | None = None,
        isaac_body_names: list[str] | None = None,
        body_aliases: dict[str, str] | None = None,
        post_ik_ground_bodies: list[str] | None = None,
        post_ik_ground_clearance: float = 0.0,
        post_ik_ground_mode: str = "lift_only",
        post_ik_foot_floor_barrier: dict | None = None,
        temporal_posture_cost: float = 0.0,
        max_output_joint_velocity_rad_s: float = 0.0,
        max_output_joint_acceleration_rad_s2: float = 0.0,
        max_output_joint_jerk_rad_s3: float = 0.0,
        postprocess_joint_gaussian_sigma_frames: float = 0.0,
        postprocess_support_projection: bool = False,
        postprocess_min_active_support_height_m: float = -0.009,
        postprocess_max_stable_support_height_above_clearance_m: float = 0.018,
    ):
        self.xml_file = model_path
        self.keypoint_path = keypoint_path
        self.ik_match_table = ik_match_table
        self.model = self._load_model_with_ground(self.xml_file)
        self.data = mujoco.MjData(self.model)
        self.max_iter = int(max_ik_iterations)
        if self.max_iter < 0:
            raise ValueError(f"max_ik_iterations must be >= 0, got {self.max_iter}")
        self.initial_settle_iterations = int(initial_settle_iterations)
        if not 0 <= self.initial_settle_iterations <= self.max_iter:
            raise ValueError(
                "initial_settle_iterations 必须位于 [0,max_ik_iterations]: "
                f"actual={self.initial_settle_iterations}, max={self.max_iter}"
            )
        self.ik_error_improvement_threshold = float(ik_error_improvement_threshold)
        if self.ik_error_improvement_threshold < 0.0:
            raise ValueError(
                "ik_error_improvement_threshold must be >= 0, got "
                f"{self.ik_error_improvement_threshold}"
            )
        self.ik_error_metric = str(ik_error_metric)
        if self.ik_error_metric not in {"legacy_raw", "solver_weighted"}:
            raise ValueError(
                "ik_error_metric must be legacy_raw or solver_weighted, got "
                f"{self.ik_error_metric}"
            )
        self.damping = damping
        self.render_debug = render_debug
        self.joints_limit_offset_degrees = joints_limit_offset_degrees or {}
        self.contact_body_names = self._normalize_contact_body_names(contact_body_names)
        self.contact_position_cost = float(contact_position_cost)
        self.contact_map_config = contact_map
        self.contact_task_mode = str(contact_task_mode)
        if self.contact_task_mode not in {
            "legacy_hold",
            "active_only",
            "paired_support",
        }:
            raise ValueError(
                "contact_task_mode must be legacy_hold, active_only or "
                "paired_support, got "
                f"{self.contact_task_mode!r}"
            )
        self.contact_position_aggregation = str(contact_position_aggregation)
        if self.contact_position_aggregation not in {"mean", "first", "mean_ramp"}:
            raise ValueError(
                "contact_position_aggregation must be 'mean', 'first' or 'mean_ramp', got "
                f"{self.contact_position_aggregation!r}"
            )
        self.contact_transition_window_seconds = float(
            contact_transition_window_seconds
        )
        if (
            not np.isfinite(self.contact_transition_window_seconds)
            or self.contact_transition_window_seconds < 0.0
        ):
            raise ValueError(
                "contact_transition_window_seconds must be non-negative and finite, got "
                f"{self.contact_transition_window_seconds}"
            )
        self.contact_weight_ramp_seconds = float(contact_weight_ramp_seconds)
        if (
            not np.isfinite(self.contact_weight_ramp_seconds)
            or self.contact_weight_ramp_seconds < 0.0
        ):
            raise ValueError(
                "contact_weight_ramp_seconds must be non-negative and finite, got "
                f"{self.contact_weight_ramp_seconds}"
            )
        self.paired_flat_orientation_cost = float(paired_flat_orientation_cost)
        if (
            not np.isfinite(self.paired_flat_orientation_cost)
            or self.paired_flat_orientation_cost < 0.0
        ):
            raise ValueError(
                "paired_flat_orientation_cost must be non-negative and finite, got "
                f"{self.paired_flat_orientation_cost}"
            )
        self.heel_toe_height_difference_cost = float(
            heel_toe_height_difference_cost
        )
        if (
            not np.isfinite(self.heel_toe_height_difference_cost)
            or self.heel_toe_height_difference_cost < 0.0
        ):
            raise ValueError(
                "heel_toe_height_difference_cost must be non-negative and finite, got "
                f"{self.heel_toe_height_difference_cost}"
            )
        self.heel_toe_height_difference_ramp_seconds = float(
            heel_toe_height_difference_ramp_seconds
        )
        if (
            not np.isfinite(self.heel_toe_height_difference_ramp_seconds)
            or self.heel_toe_height_difference_ramp_seconds < 0.0
        ):
            raise ValueError(
                "heel_toe_height_difference_ramp_seconds must be non-negative "
                f"and finite, got {self.heel_toe_height_difference_ramp_seconds}"
            )
        self.initial_root_pose = initial_root_pose
        self.initial_joint_positions = initial_joint_positions or {}
        self.robot_name = robot_name
        self.config_path = config_path
        self.isaac_joint_names = list(isaac_joint_names or [])
        self.isaac_body_names = list(isaac_body_names or [])
        self.body_aliases = dict(body_aliases or {})
        self.post_ik_ground_bodies = list(post_ik_ground_bodies or [])
        self.post_ik_ground_clearance = float(post_ik_ground_clearance)
        if self.post_ik_ground_clearance < 0.0 or not np.isfinite(
            self.post_ik_ground_clearance
        ):
            raise ValueError(
                "post_ik_ground_clearance must be non-negative and finite, got "
                f"{self.post_ik_ground_clearance}"
            )
        self.post_ik_ground_mode = str(post_ik_ground_mode)
        if self.post_ik_ground_mode not in {"lift_only", "support_project"}:
            raise ValueError(
                "post_ik_ground_mode must be lift_only or support_project, got "
                f"{self.post_ik_ground_mode!r}"
            )
        self.post_ik_foot_floor_barrier_config = dict(
            post_ik_foot_floor_barrier or {}
        )
        self.temporal_posture_cost = float(temporal_posture_cost)
        if not np.isfinite(self.temporal_posture_cost) or self.temporal_posture_cost < 0.0:
            raise ValueError(
                "temporal_posture_cost must be non-negative and finite, got "
                f"{self.temporal_posture_cost}"
            )
        self.max_output_joint_velocity_rad_s = float(
            max_output_joint_velocity_rad_s
        )
        if (
            not np.isfinite(self.max_output_joint_velocity_rad_s)
            or self.max_output_joint_velocity_rad_s < 0.0
        ):
            raise ValueError(
                "max_output_joint_velocity_rad_s must be non-negative and finite, got "
                f"{self.max_output_joint_velocity_rad_s}"
            )
        self.max_output_joint_acceleration_rad_s2 = float(
            max_output_joint_acceleration_rad_s2
        )
        self.max_output_joint_jerk_rad_s3 = float(max_output_joint_jerk_rad_s3)
        for name, value in (
            (
                "max_output_joint_acceleration_rad_s2",
                self.max_output_joint_acceleration_rad_s2,
            ),
            ("max_output_joint_jerk_rad_s3", self.max_output_joint_jerk_rad_s3),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite, got {value}")
        self.output_joint_velocity_state = None
        self.output_joint_acceleration_state = None
        self.postprocess_joint_gaussian_sigma_frames = float(
            postprocess_joint_gaussian_sigma_frames
        )
        if (
            not np.isfinite(self.postprocess_joint_gaussian_sigma_frames)
            or self.postprocess_joint_gaussian_sigma_frames < 0.0
        ):
            raise ValueError(
                "postprocess_joint_gaussian_sigma_frames must be non-negative "
                f"and finite, got {self.postprocess_joint_gaussian_sigma_frames}"
            )
        self.postprocess_support_projection = bool(postprocess_support_projection)
        self.postprocess_min_active_support_height_m = float(
            postprocess_min_active_support_height_m
        )
        self.postprocess_max_stable_support_height_above_clearance_m = float(
            postprocess_max_stable_support_height_above_clearance_m
        )
        if not np.isfinite(self.postprocess_min_active_support_height_m):
            raise ValueError(
                "postprocess_min_active_support_height_m must be finite"
            )
        if (
            not np.isfinite(
                self.postprocess_max_stable_support_height_above_clearance_m
            )
            or self.postprocess_max_stable_support_height_above_clearance_m < 0.0
        ):
            raise ValueError(
                "postprocess_max_stable_support_height_above_clearance_m must "
                "be non-negative and finite"
            )
        self.postprocess_statistics = {}
        self.foot_floor_barrier_velocity_state = {}
        self.foot_floor_barrier_acceleration_state = {}
        self.temporal_posture_task = None
        self.verbose = verbose
        self.solver = solver
        self.human_body_to_task = {}
        self.task_errors = {}
        self.result_pos = []
        self.contact_state_name_to_idx = {}
        self.keypoint_name_to_idx = {}
        self.body_name_to_source_keypoint = {}
        self.body_name_to_contact_task = {}
        self.contact_targets = []
        self.current_contact_points = []
        self.frame_final_errors = []
        self.frame_iteration_counts = []
        self.frame_ground_corrections = []
        self.frame_floor_barrier_adjustment_counts = []
        self.frame_floor_barrier_max_adjustments = []
        self.frame_output_velocity_clip_counts = []
        self.frame_raw_max_joint_velocities = []
        self.current_ground_body_ids = []
        self.paired_support_summary = {}
        self.paired_orientation_targets = []
        self.paired_precontact_targets = []
        self.heel_toe_height_targets = []
        self.heel_toe_height_summary = {}
        self.keypoints_metadata = {}

        self.robot_motor_names = {}

        self.setup_retarget_configuration()
        self._setup_post_ik_ground_clearance()
        self.load_keypoints()
        self._validate_ik_keypoints()
        self.setup_contact_targets()

    def _normalize_contact_body_names(self, contact_body_names):
        if contact_body_names is None:
            return []
        if isinstance(contact_body_names, dict):
            flattened_body_names = []
            for field_name, contact_names in self.LEGACY_CONTACT_CONFIG_TO_NAMES.items():
                body_names = contact_body_names.get(field_name, [])
                if len(body_names) != len(contact_names):
                    continue
                flattened_body_names.extend(body_names)
            return flattened_body_names
        if isinstance(contact_body_names, (list, tuple)):
            return list(contact_body_names)
        raise TypeError(
            "contact_body_names must be a list/tuple of body names or a legacy contact-body mapping"
        )

    def _load_model_with_ground(self, xml_file: str) -> mujoco.MjModel:
        """Load the robot XML and add a MuJoCo default-style ground (checker plane + skybox).

        The ground geom has collision disabled (contype/conaffinity=0) and serves only as a
        visual reference; it does not affect mink IK's pure kinematic solving.
        """
        spec = mujoco.MjSpec.from_file(xml_file)

        # Skip if ground/skybox assets already exist to avoid name conflicts
        existing_tex = {t.name for t in spec.textures}
        existing_mat = {m.name for m in spec.materials}
        existing_geoms = {g.name: g for g in spec.geoms if g.name}

        if "skybox" not in existing_tex:
            spec.add_texture(
                name="skybox",
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                rgb1=[0.3, 0.5, 0.7],
                rgb2=[0.0, 0.0, 0.0],
                width=512,
                height=512,
            )
        if "groundplane" not in existing_tex:
            spec.add_texture(
                name="groundplane",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                rgb1=[0.2, 0.3, 0.4],
                rgb2=[0.1, 0.2, 0.3],
                width=300,
                height=300,
            )
        if "groundplane" not in existing_mat:
            spec.add_material(
                name="groundplane",
                textures=["", "groundplane"],
                texrepeat=[5, 5],
                texuniform=True,
                reflectance=0.2,
            )

        # Lighting: main directional light + angled fill light to avoid an overly dark scene
        spec.worldbody.add_light(
            pos=[0, 0, 20.0],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.7, 0.7, 0.7],
            specular=[0.3, 0.3, 0.3],
        )
        spec.worldbody.add_light(
            pos=[4, 4, 6.0],
            dir=[-0.5, -0.5, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.4, 0.4, 0.4],
            specular=[0.1, 0.1, 0.1],
        )

        # Reuse an existing ground. Models without one still receive a visual plane.
        ground = existing_geoms.get("ground")
        if ground is None:
            ground = spec.worldbody.add_geom(
                name="ground",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[0, 0, 0.05],
                material="groundplane",
                pos=[0, 0, 0],
            )
        ground.contype = 0
        ground.conaffinity = 0

        return spec.compile()

    def load_keypoints(self):
        with open(self.keypoint_path, "rb") as f:
                keypoints_data = pickle.load(f)
        self.keypoint_names = keypoints_data["keypoint_names"]
        self.keypoint_name_to_idx = {
            keypoint_name: idx for idx, keypoint_name in enumerate(self.keypoint_names)
        }
        self.keypoints_pos = keypoints_data["positions"] 
        self.keypoints_quat = keypoints_data["quaternions"]  
        self.contact_names = keypoints_data.get("contact_names", [])
        self.contact_state_name_to_idx = {
            contact_name: idx for idx, contact_name in enumerate(self.contact_names)
        }
        self.contact_seq = keypoints_data["contact_states"]
        self.fps = keypoints_data.get("fps", 30)
        self.time_step = 1.0 / self.fps
        self.num_frames = self.keypoints_pos.shape[0]
        self.num_keypoints = self.keypoints_pos.shape[1]
        self.keypoints_metadata = {
            key: value
            for key, value in keypoints_data.items()
            if key
            not in {
                "positions",
                "quaternions",
                "contact_positions",
                "contact_speeds",
                "contact_states",
            }
        }

        if self.keypoints_pos.shape != (self.num_frames, self.num_keypoints, 3):
            raise ValueError(f"Invalid keypoint position shape: {self.keypoints_pos.shape}")
        if self.keypoints_quat.shape != (self.num_frames, self.num_keypoints, 4):
            raise ValueError(f"Invalid keypoint quaternion shape: {self.keypoints_quat.shape}")
        if self.contact_seq.shape != (self.num_frames, len(self.contact_names)):
            raise ValueError(
                "Invalid contact state shape: expected "
                f"{(self.num_frames, len(self.contact_names))}, got {self.contact_seq.shape}"
            )
        if not np.all(np.isfinite(self.keypoints_pos)) or not np.all(
            np.isfinite(self.keypoints_quat)
        ):
            raise ValueError(f"Keypoints contain NaN/Inf: {self.keypoint_path}")

    def _validate_ik_keypoints(self):
        missing = sorted(set(self.human_body_to_task) - set(self.keypoint_names))
        if missing:
            raise ValueError(
                f"IK source keypoints are missing in {self.keypoint_path}: {missing}"
            )

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        self.tasks = []
        self.robot_frame_names = []  # Robot body name corresponding to each task

        # Apply offsets to the configured joint limits: raise the lower bound
        self._apply_joints_limit_offset()
        self._apply_initial_configuration()
        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        # VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
        # self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 

        for keypoint_name, entry in self.ik_match_table.items():
            robot_frame, pos_weight, rot_weight = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=robot_frame,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                # Use the keypoint name as the key for indexing keypoint data in update_targets
                self.human_body_to_task[keypoint_name] = task
                self.body_name_to_source_keypoint[robot_frame] = keypoint_name

                self.tasks.append(task)
                self.robot_frame_names.append(robot_frame)
                self.task_errors[task] = []
        self.base_tasks = list(self.tasks)
        if self.temporal_posture_cost > 0.0:
            self.temporal_posture_task = mink.PostureTask(
                self.model,
                cost=self.temporal_posture_cost,
                lm_damping=1.0,
            )
            self.temporal_posture_task.set_target_from_configuration(
                self.configuration
            )
            self.base_tasks.append(self.temporal_posture_task)

    def _apply_initial_configuration(self):
        """Apply optional free-joint and scalar-joint qpos values."""
        if not self.initial_root_pose and not self.initial_joint_positions:
            return

        qpos = np.asarray(self.model.qpos0, dtype=np.float64).copy()
        free_joint_ids = [
            joint_id
            for joint_id in range(self.model.njnt)
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if self.initial_root_pose:
            if len(free_joint_ids) != 1:
                raise ValueError(
                    "initial_root_pose requires exactly one freejoint, got "
                    f"{len(free_joint_ids)} in {self.xml_file}"
                )
            position = np.asarray(
                self.initial_root_pose.get("position"), dtype=np.float64
            )
            quaternion = np.asarray(
                self.initial_root_pose.get("quaternion_wxyz"), dtype=np.float64
            )
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(f"initial_root_pose.position must be finite [3], got {position}")
            if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
                raise ValueError(
                    "initial_root_pose.quaternion_wxyz must be finite [4], got "
                    f"{quaternion}"
                )
            norm = float(np.linalg.norm(quaternion))
            if norm <= 1.0e-12:
                raise ValueError("initial_root_pose.quaternion_wxyz has zero norm")
            qadr = int(self.model.jnt_qposadr[free_joint_ids[0]])
            qpos[qadr : qadr + 3] = position
            qpos[qadr + 3 : qadr + 7] = quaternion / norm

        for joint_name, value in self.initial_joint_positions.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name)
            )
            if joint_id < 0:
                raise ValueError(f"Initial joint not found: {joint_name} in {self.xml_file}")
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(f"Initial joint must be scalar: {joint_name}, type={joint_type}")
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError(f"Initial joint value must be finite: {joint_name}={value}")
            if bool(self.model.jnt_limited[joint_id]):
                lower, upper = self.model.jnt_range[joint_id]
                if scalar < lower - 1.0e-9 or scalar > upper + 1.0e-9:
                    raise ValueError(
                        f"Initial joint outside tightened limit: {joint_name}={scalar}, "
                        f"range=[{lower}, {upper}]"
                    )
            qpos[int(self.model.jnt_qposadr[joint_id])] = scalar

        self.configuration.update(qpos)
        mujoco.mj_forward(self.model, self.configuration.data)

    def _setup_post_ik_ground_clearance(self):
        """Resolve optional foot-marker IDs and the sole freejoint z address."""
        self.post_ik_ground_body_ids = []
        self.post_ik_root_z_address = None
        self.foot_floor_barrier_specs = []
        if not self.post_ik_ground_bodies:
            return
        for body_name in self.post_ik_ground_bodies:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                raise ValueError(f"post_ik_ground_bodies body not found: {body_name}")
            self.post_ik_ground_body_ids.append(body_id)
        free_joint_ids = [
            joint_id
            for joint_id in range(self.model.njnt)
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free_joint_ids) != 1:
            raise ValueError(
                "post-IK ground clearance requires exactly one freejoint, got "
                f"{len(free_joint_ids)}"
            )
        self.post_ik_root_z_address = int(
            self.model.jnt_qposadr[free_joint_ids[0]] + 2
        )
        barrier_config = self.post_ik_foot_floor_barrier_config
        if not bool(barrier_config.get("enabled", False)):
            return
        feet_config = barrier_config.get("feet", {})
        if not isinstance(feet_config, dict) or not feet_config:
            raise ValueError(
                "post_ik_foot_floor_barrier.feet 必须是非空映射"
            )
        self.foot_floor_barrier_grid_samples = int(
            barrier_config.get("grid_samples", 33)
        )
        self.foot_floor_barrier_bisection_iterations = int(
            barrier_config.get("bisection_iterations", 16)
        )
        self.foot_floor_barrier_anticipation_frames = int(
            barrier_config.get("anticipation_frames", 0)
        )
        self.foot_floor_barrier_max_velocity = float(
            barrier_config.get("max_velocity_rad_s", 0.0)
        )
        self.foot_floor_barrier_max_acceleration = float(
            barrier_config.get("max_acceleration_rad_s2", 0.0)
        )
        self.foot_floor_barrier_max_jerk = float(
            barrier_config.get("max_jerk_rad_s3", 0.0)
        )
        if self.foot_floor_barrier_grid_samples < 3:
            raise ValueError("foot floor barrier grid_samples 必须 >= 3")
        if self.foot_floor_barrier_bisection_iterations < 1:
            raise ValueError("foot floor barrier bisection_iterations 必须 >= 1")
        if self.foot_floor_barrier_anticipation_frames < 0:
            raise ValueError("foot floor barrier anticipation_frames 必须 >= 0")
        for name, value in (
            ("max_velocity_rad_s", self.foot_floor_barrier_max_velocity),
            ("max_acceleration_rad_s2", self.foot_floor_barrier_max_acceleration),
            ("max_jerk_rad_s3", self.foot_floor_barrier_max_jerk),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"foot floor barrier {name} 必须为正有限值")
        for side, entry in feet_config.items():
            if not isinstance(entry, dict):
                raise ValueError(f"foot floor barrier {side} 必须是映射")
            ankle_joint_name = str(entry.get("ankle_pitch_joint", ""))
            marker_names = [
                str(entry.get("heel_body", "")),
                str(entry.get("toe_body", "")),
            ]
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, ankle_joint_name
            )
            body_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in marker_names
            ]
            if joint_id < 0 or any(body_id < 0 for body_id in body_ids):
                raise ValueError(
                    f"foot floor barrier {side} 引用不存在: "
                    f"joint={ankle_joint_name}, bodies={marker_names}"
                )
            if int(self.model.jnt_type[joint_id]) not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(
                    f"foot floor barrier 只支持单自由度踝 pitch: {ankle_joint_name}"
                )
            self.foot_floor_barrier_specs.append(
                {
                    "side": str(side),
                    "joint_name": ankle_joint_name,
                    "qpos_address": int(self.model.jnt_qposadr[joint_id]),
                    "joint_range": np.asarray(
                        self.model.jnt_range[joint_id], dtype=np.float64
                    ).copy(),
                    "body_ids": body_ids,
                    "heel_contact_name": str(
                        entry.get("heel_contact", f"{side}_foot_end")
                    ),
                    "toe_contact_name": str(
                        entry.get("toe_contact", f"{side}_toe")
                    ),
                }
            )

    def _apply_post_ik_ground_clearance(self) -> float:
        """按配置修正根节点高度，使应当支撑的足点落到目标地面。

        ``lift_only`` 保留历史行为：四个足点中任何一点穿地时只向上抬根节点。
        ``support_project`` 则只查看当前离散支撑相位要求激活的足点，并允许有符号
        修正；因此双点平足支撑时脚跟、脚尖会一起落到 clearance，高空悬脚也会被
        向下放回地面，而摆动期不会错误地把整机吸向地面。
        """
        if not self.post_ik_ground_body_ids:
            return 0.0
        body_ids = self.post_ik_ground_body_ids
        if self.post_ik_ground_mode == "support_project":
            body_ids = self.current_ground_body_ids
            if not body_ids:
                return 0.0
            heights = np.asarray(
                self.configuration.data.xpos[body_ids, 2], dtype=np.float64
            )
            support_height = float(np.min(heights))
            correction = self.post_ik_ground_clearance - support_height
        else:
            minimum_height = float(
                np.min(self.configuration.data.xpos[body_ids, 2])
            )
            correction = self.post_ik_ground_clearance - minimum_height
            correction = max(0.0, correction)
        if abs(correction) > 0.0:
            qpos = self.configuration.data.qpos.copy()
            qpos[self.post_ik_root_z_address] += correction
            self.configuration.update(qpos)
            mujoco.mj_forward(self.model, self.configuration.data)
        return correction

    def _apply_foot_floor_barrier(
        self, frame_idx: int, previous_output_qpos: np.ndarray
    ) -> tuple[int, float]:
        """用踝 pitch 的最小残差改变量阻止前后足点穿过可视地板。

        MuJoCo 地板在重定向模型中仅用于显示，不参与 Mink 的碰撞约束；摆动脚因而
        可能有毫米级残余穿地。主 IK 已通过接触前竖直任务让整条腿提前准备落地；
        这里仅对残差做一维有界搜索，只改变该脚踝 pitch，并选择离当前角度最近、
        能让所需足点回到地面以上的解。搜索不修改根节点、不锁摆动脚 XY，也不作用
        于本来就在地面以上的帧，因此不会再引入事后整腿投影造成的膝关节 jerk。
        """
        if not self.foot_floor_barrier_specs:
            return 0, 0.0

        adjustment_count = 0
        maximum_adjustment = 0.0
        clearance = self.post_ik_ground_clearance
        contact_targets_by_name = {
            item["contact_name"]: item for item in self.contact_targets
        }
        for spec in self.foot_floor_barrier_specs:
            heel_target = contact_targets_by_name.get(spec["heel_contact_name"])
            toe_target = contact_targets_by_name.get(spec["toe_contact_name"])
            if heel_target is None or toe_target is None:
                raise ValueError(
                    "foot floor barrier 缺少成对接触目标: "
                    f"{spec['heel_contact_name']}/{spec['toe_contact_name']}"
                )
            lookahead_end = min(
                self.num_frames,
                frame_idx + self.foot_floor_barrier_anticipation_frames + 1,
            )
            heel_window = np.asarray(
                heel_target["paired_ground_active"][frame_idx:lookahead_end],
                dtype=bool,
            )
            toe_window = np.asarray(
                toe_target["paired_ground_active"][frame_idx:lookahead_end],
                dtype=bool,
            )
            heel_expected = bool(np.any(heel_window))
            toe_expected = bool(np.any(toe_window))
            heel_weight = float(heel_target["paired_weights"][frame_idx])
            toe_weight = float(toe_target["paired_weights"][frame_idx])
            heel_transition = heel_expected and heel_weight < 1.0 - 1.0e-9
            toe_transition = toe_expected and toe_weight < 1.0 - 1.0e-9
            # 权重到 1 后，成对接触任务与稳定根投影已经负责贴地；屏障只覆盖
            # “接触即将发生或正在渐入”的空窗，避免在长支撑段反复改写踝 pitch。
            heel_active_now = bool(heel_window[0])
            toe_active_now = bool(toe_window[0])
            if heel_active_now and heel_transition:
                body_ids = spec["body_ids"]
            elif toe_active_now and toe_transition:
                body_ids = [spec["body_ids"][1]]
            elif not heel_active_now and not toe_active_now:
                heel_offset = (
                    int(np.flatnonzero(heel_window)[0])
                    if heel_expected
                    else self.foot_floor_barrier_anticipation_frames + 1
                )
                toe_offset = (
                    int(np.flatnonzero(toe_window)[0])
                    if toe_expected
                    else self.foot_floor_barrier_anticipation_frames + 1
                )
                if toe_transition and toe_offset < heel_offset:
                    body_ids = [spec["body_ids"][1]]
                elif heel_transition:
                    body_ids = spec["body_ids"]
                else:
                    continue
            else:
                continue
            current_min_height = float(
                np.min(self.configuration.data.xpos[body_ids, 2])
            )
            if current_min_height >= clearance - 1.0e-9:
                continue

            original_qpos = self.configuration.data.qpos.copy()
            address = int(spec["qpos_address"])
            original_angle = float(original_qpos[address])
            lower, upper = spec["joint_range"]
            sample_angles = np.linspace(
                float(lower),
                float(upper),
                self.foot_floor_barrier_grid_samples,
            )
            sample_heights = []
            for angle in sample_angles:
                candidate = original_qpos.copy()
                candidate[address] = float(angle)
                self.configuration.update(candidate)
                sample_heights.append(
                    float(np.min(self.configuration.data.xpos[body_ids, 2]))
                )
            sample_heights = np.asarray(sample_heights, dtype=np.float64)
            feasible_indices = np.flatnonzero(sample_heights >= clearance)
            if feasible_indices.size:
                selected_idx = int(
                    feasible_indices[
                        np.argmin(
                            np.abs(sample_angles[feasible_indices] - original_angle)
                        )
                    ]
                )
                feasible_angle = float(sample_angles[selected_idx])
                infeasible_angle = original_angle
                for _ in range(self.foot_floor_barrier_bisection_iterations):
                    midpoint = 0.5 * (infeasible_angle + feasible_angle)
                    candidate = original_qpos.copy()
                    candidate[address] = midpoint
                    self.configuration.update(candidate)
                    midpoint_height = float(
                        np.min(self.configuration.data.xpos[body_ids, 2])
                    )
                    if midpoint_height >= clearance:
                        feasible_angle = midpoint
                    else:
                        infeasible_angle = midpoint
                selected_angle = feasible_angle
            else:
                # 极端姿态下单独转踝无法让前后点同时离地，此时仍取地板余量最大的
                # 有界角度；最终硬验收会明确拒绝剩余穿透，不会静默放行。
                selected_angle = float(sample_angles[int(np.argmax(sample_heights))])

            corrected_qpos = original_qpos.copy()
            corrected_qpos[address] = selected_angle
            self.configuration.update(corrected_qpos)
            adjustment = abs(selected_angle - original_angle)
            adjustment_count += int(adjustment > 1.0e-12)
            maximum_adjustment = max(maximum_adjustment, adjustment)

        # 屏障进入和退出都经过同一组三阶状态约束。即使本帧没有穿地，也要继续
        # 对原 IK 踝角做限幅，否则屏障解除帧会立刻跳回未约束姿态。
        limited_qpos = self.configuration.data.qpos.copy()
        previous_output_qpos = np.asarray(previous_output_qpos, dtype=np.float64)
        for spec in self.foot_floor_barrier_specs:
            address = int(spec["qpos_address"])
            state_key = str(spec["side"])
            if frame_idx == 0:
                self.foot_floor_barrier_velocity_state[state_key] = 0.0
                self.foot_floor_barrier_acceleration_state[state_key] = 0.0
                continue
            desired_velocity = (
                limited_qpos[address] - previous_output_qpos[address]
            ) * self.fps
            previous_velocity = float(
                self.foot_floor_barrier_velocity_state.get(state_key, 0.0)
            )
            previous_acceleration = float(
                self.foot_floor_barrier_acceleration_state.get(state_key, 0.0)
            )
            desired_acceleration = (
                desired_velocity - previous_velocity
            ) * self.fps
            jerk_step = self.foot_floor_barrier_max_jerk / self.fps
            acceleration = float(
                np.clip(
                    desired_acceleration,
                    previous_acceleration - jerk_step,
                    previous_acceleration + jerk_step,
                )
            )
            acceleration = float(
                np.clip(
                    acceleration,
                    -self.foot_floor_barrier_max_acceleration,
                    self.foot_floor_barrier_max_acceleration,
                )
            )
            velocity = previous_velocity + acceleration / self.fps
            velocity = float(
                np.clip(
                    velocity,
                    -self.foot_floor_barrier_max_velocity,
                    self.foot_floor_barrier_max_velocity,
                )
            )
            lower, upper = spec["joint_range"]
            limited_angle = float(
                np.clip(
                    previous_output_qpos[address] + velocity / self.fps,
                    lower,
                    upper,
                )
            )
            actual_velocity = (
                limited_angle - previous_output_qpos[address]
            ) * self.fps
            actual_acceleration = (
                actual_velocity - previous_velocity
            ) * self.fps
            limited_qpos[address] = limited_angle
            self.foot_floor_barrier_velocity_state[state_key] = actual_velocity
            self.foot_floor_barrier_acceleration_state[state_key] = (
                actual_acceleration
            )
        self.configuration.update(limited_qpos)
        return adjustment_count, maximum_adjustment

    def _apply_output_joint_velocity_limit(
        self, previous_qpos: np.ndarray
    ) -> tuple[int, float]:
        """在实际输出 fps 边界限制关节速度、加速度和 jerk，并返回审计统计。

        Mink 内部一次输出帧会做多次小步迭代，单次求解的 velocity limit 不能限制
        这些迭代累加后的最终帧间速度。本函数因此在所有 IK 迭代结束后、根高度
        投影前，按真实动作 ``fps`` 对有限位标量关节做最终三阶限幅。加速度由
        上一输出速度递推，jerk 再限制加速度变化，所以限速本身也不会制造新的
        单帧尖峰。freejoint 不在此处修改，避免把地面投影和根运动合同混进关节
        限幅。
        """
        if self.max_output_joint_velocity_rad_s <= 0.0:
            return 0, 0.0
        previous_qpos = np.asarray(previous_qpos, dtype=np.float64)
        current_qpos = self.configuration.data.qpos.copy()
        addresses = []
        lower_bounds = []
        upper_bounds = []
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] in {
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            }:
                addresses.append(int(self.model.jnt_qposadr[joint_id]))
                if bool(self.model.jnt_limited[joint_id]):
                    lower_bounds.append(float(self.model.jnt_range[joint_id, 0]))
                    upper_bounds.append(float(self.model.jnt_range[joint_id, 1]))
                else:
                    lower_bounds.append(-np.inf)
                    upper_bounds.append(np.inf)
        if not addresses:
            return 0, 0.0
        addresses_array = np.asarray(addresses, dtype=np.int32)
        raw_delta = current_qpos[addresses_array] - previous_qpos[addresses_array]
        desired_velocity = raw_delta * self.fps
        raw_max_velocity = float(np.max(np.abs(desired_velocity)))
        output_velocity = desired_velocity.copy()
        if self.output_joint_velocity_state is not None:
            desired_acceleration = (
                desired_velocity - self.output_joint_velocity_state
            ) * self.fps
            output_acceleration = desired_acceleration
            if (
                self.max_output_joint_jerk_rad_s3 > 0.0
                and self.output_joint_acceleration_state is not None
            ):
                maximum_acceleration_change = (
                    self.max_output_joint_jerk_rad_s3 / self.fps
                )
                output_acceleration = self.output_joint_acceleration_state + np.clip(
                    output_acceleration - self.output_joint_acceleration_state,
                    -maximum_acceleration_change,
                    maximum_acceleration_change,
                )
            if self.max_output_joint_acceleration_rad_s2 > 0.0:
                output_acceleration = np.clip(
                    output_acceleration,
                    -self.max_output_joint_acceleration_rad_s2,
                    self.max_output_joint_acceleration_rad_s2,
                )
            output_velocity = self.output_joint_velocity_state + (
                output_acceleration / self.fps
            )
        output_velocity = np.clip(
            output_velocity,
            -self.max_output_joint_velocity_rad_s,
            self.max_output_joint_velocity_rad_s,
        )
        output_positions = previous_qpos[addresses_array] + output_velocity / self.fps
        output_positions = np.clip(
            output_positions,
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        )
        # 关节限位裁剪会改变真实输出速度，后续的加速度/jerk 状态必须以实际值
        # 递推，否则下一帧会从一个不存在的速度状态继续积分并越过另一侧限位。
        output_velocity = (
            output_positions - previous_qpos[addresses_array]
        ) * self.fps
        if self.output_joint_velocity_state is None:
            actual_acceleration = np.zeros_like(output_velocity)
        else:
            actual_acceleration = (
                output_velocity - self.output_joint_velocity_state
            ) * self.fps
        clipped = np.abs(output_velocity - desired_velocity) > 1.0e-12
        clip_count = int(np.count_nonzero(clipped))
        self.output_joint_velocity_state = output_velocity.copy()
        self.output_joint_acceleration_state = actual_acceleration.copy()
        if clip_count:
            current_qpos[addresses_array] = output_positions
            self.configuration.update(current_qpos)
            mujoco.mj_forward(self.model, self.configuration.data)
        return clip_count, raw_max_velocity

    def _postprocess_result_trajectory(self) -> None:
        """对完整关节轨迹做对称平滑，并按离散支撑相位重新投影根高度。

        逐帧 IK 不知道未来目标，在线速度/加速度限幅在目标反向时可能产生制动
        过冲。完整轨迹生成后，使用时间对称的高斯核平滑所有有限位标量关节，
        不引入相位延迟；随后仅改变 freejoint 的 Z，使当前平足 heel/toe 或
        toe-off 脚尖中的最低支撑点回到 clearance。该步骤不改变根 XY、姿态或
        足点水平位置，最终速度、加速度、jerk、滑移和足底高度仍由独立验证器
        重新计算并执行硬拒绝。
        """
        if not self.result_pos:
            self.postprocess_statistics = {
                "applied": False,
                "reason": "empty_result",
            }
            return
        result = np.asarray(self.result_pos, dtype=np.float64).copy()
        scalar_addresses = []
        lower_bounds = []
        upper_bounds = []
        for joint_id in range(self.model.njnt):
            if int(self.model.jnt_type[joint_id]) not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            scalar_addresses.append(int(self.model.jnt_qposadr[joint_id]))
            if bool(self.model.jnt_limited[joint_id]):
                lower_bounds.append(float(self.model.jnt_range[joint_id, 0]))
                upper_bounds.append(float(self.model.jnt_range[joint_id, 1]))
            else:
                lower_bounds.append(-np.inf)
                upper_bounds.append(np.inf)
        addresses = np.asarray(scalar_addresses, dtype=np.int32)
        original_joint_positions = result[:, addresses].copy()
        sigma = self.postprocess_joint_gaussian_sigma_frames
        if sigma > 0.0 and result.shape[0] > 1 and addresses.size:
            result[:, addresses] = gaussian_filter1d(
                result[:, addresses], sigma=sigma, axis=0, mode="nearest"
            )
            result[:, addresses] = np.clip(
                result[:, addresses],
                np.asarray(lower_bounds, dtype=np.float64),
                np.asarray(upper_bounds, dtype=np.float64),
            )

        root_corrections = np.zeros(result.shape[0], dtype=np.float64)
        if self.postprocess_support_projection:
            if self.post_ik_root_z_address is None:
                raise ValueError(
                    "postprocess_support_projection requires one resolved freejoint"
                )
            body_ids_by_target = {
                item["contact_name"]: mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    item["body_name"],
                )
                for item in self.contact_targets
            }
            for frame_idx in range(result.shape[0]):
                self.configuration.update(result[frame_idx])
                active_targets = [
                    item
                    for item in self.contact_targets
                    if bool(item["paired_ground_active"][frame_idx])
                ]
                support_body_ids = [
                    body_ids_by_target[item["contact_name"]]
                    for item in active_targets
                ]
                if not support_body_ids:
                    continue
                support_heights = np.asarray(
                    self.configuration.data.xpos[support_body_ids, 2],
                    dtype=np.float64,
                )
                preferred_correction = (
                    self.post_ik_ground_clearance
                    - float(np.min(support_heights))
                )
                minimum_correction = (
                    self.postprocess_min_active_support_height_m
                    - float(np.min(support_heights))
                )
                stable_body_ids = [
                    body_ids_by_target[item["contact_name"]]
                    for item in active_targets
                    if float(item["paired_weights"][frame_idx])
                    >= 1.0 - 1.0e-9
                ]
                maximum_correction = np.inf
                if stable_body_ids:
                    stable_maximum_height = float(
                        np.max(
                            self.configuration.data.xpos[stable_body_ids, 2]
                        )
                    )
                    maximum_correction = (
                        self.post_ik_ground_clearance
                        + self.postprocess_max_stable_support_height_above_clearance_m
                        - stable_maximum_height
                    )
                if minimum_correction <= maximum_correction:
                    correction = float(
                        np.clip(
                            preferred_correction,
                            minimum_correction,
                            maximum_correction,
                        )
                    )
                else:
                    # 根 Z 无法同时满足两侧边界时优先保护已稳定支撑脚；最终验证
                    # 会以非零退出码拒绝仍然穿地的动作，不会静默放行。
                    correction = float(maximum_correction)
                result[frame_idx, self.post_ik_root_z_address] += correction
                root_corrections[frame_idx] = correction

        joint_adjustments = result[:, addresses] - original_joint_positions
        self.result_pos = [row.copy() for row in result]
        self.configuration.update(result[-1])
        self.postprocess_statistics = {
            "applied": bool(
                sigma > 0.0 or self.postprocess_support_projection
            ),
            "joint_gaussian_sigma_frames": float(sigma),
            "support_projection_enabled": self.postprocess_support_projection,
            "minimum_active_support_height_m": float(
                self.postprocess_min_active_support_height_m
            ),
            "maximum_stable_support_height_above_clearance_m": float(
                self.postprocess_max_stable_support_height_above_clearance_m
            ),
            "maximum_joint_adjustment_rad": (
                float(np.max(np.abs(joint_adjustments)))
                if joint_adjustments.size
                else 0.0
            ),
            "projected_frame_count": int(
                np.count_nonzero(np.abs(root_corrections) > 1.0e-12)
            ),
            "maximum_root_z_correction_m": (
                float(np.max(np.abs(root_corrections)))
                if root_corrections.size
                else 0.0
            ),
        }

    def setup_contact_targets(self):
        if not self.contact_state_name_to_idx:
            return

        if self.contact_map_config is not None:
            mappings = self._parse_explicit_contact_map(self.contact_map_config)
        else:
            mappings = self._build_legacy_contact_mappings()

        for mapping in mappings:
            task = self._ensure_contact_task(mapping.frame_name, mapping.position_cost)
            self.contact_targets.append(
                {
                    "body_name": mapping.frame_name,
                    "contact_name": mapping.source_contact_name,
                    "contact_state_idx": self.contact_state_name_to_idx[
                        mapping.source_contact_name
                    ],
                    "source_keypoint_name": mapping.source_keypoint_name,
                    "lock_positions": self._build_contact_lock_positions(
                        contact_state_idx=self.contact_state_name_to_idx[
                            mapping.source_contact_name
                        ],
                        source_keypoint_name=mapping.source_keypoint_name,
                    ),
                    "task": task,
                    "base_position_cost": float(mapping.position_cost),
                }
            )
            if self.contact_task_mode in {"legacy_hold", "paired_support"}:
                self.tasks.append(task)
                self.robot_frame_names.append(mapping.frame_name)

        if self.contact_task_mode == "paired_support":
            self._setup_paired_support_plan()

        if self.heel_toe_height_difference_cost > 0.0:
            self._setup_heel_toe_height_difference_tasks()

        if self.verbose and self.contact_targets:
            summary = []
            for item in self.contact_targets:
                states = self.contact_seq[:, item["contact_state_idx"]].astype(bool)
                starts = states & ~np.r_[False, states[:-1]]
                summary.append(
                    f"{item['contact_name']}->{item['body_name']} "
                    f"(active_frames={int(states.sum())}, intervals={int(starts.sum())})"
                )
            print(f"[contact target] mode={self.contact_task_mode}: {summary}")

    def _setup_heel_toe_height_difference_tasks(self) -> None:
        """为左右脚构造同时接触期的 heel/toe 等高软任务与渐变权重。"""
        by_name = {item["contact_name"]: item for item in self.contact_targets}
        required = {
            "left_foot_end",
            "left_toe",
            "right_foot_end",
            "right_toe",
        }
        missing = sorted(required - set(by_name))
        if missing:
            raise ValueError(
                "heel/toe 高度差软约束需要左右脚四个接触映射: "
                f"missing={missing}"
            )

        ramp_frames = max(
            1,
            int(round(self.heel_toe_height_difference_ramp_seconds * self.fps)),
        )
        summary = {
            "base_cost": self.heel_toe_height_difference_cost,
            "ramp_seconds": self.heel_toe_height_difference_ramp_seconds,
            "ramp_frames": ramp_frames,
            "feet": {},
        }
        for side in ("left", "right"):
            heel = by_name[f"{side}_foot_end"]
            toe = by_name[f"{side}_toe"]
            heel_states = self.contact_seq[:, heel["contact_state_idx"]].astype(bool)
            toe_states = self.contact_seq[:, toe["contact_state_idx"]].astype(bool)
            flat_contact_states = heel_states & toe_states
            weights = self._ramp_binary_weights(flat_contact_states, ramp_frames)
            task = HeelToeHeightDifferenceTask(
                model=self.model,
                heel_body_name=heel["body_name"],
                toe_body_name=toe["body_name"],
                cost=0.0,
                lm_damping=1.0,
            )
            target = {
                "side": side,
                "heel_body_name": heel["body_name"],
                "toe_body_name": toe["body_name"],
                "task": task,
                "base_cost": self.heel_toe_height_difference_cost,
                "flat_contact_states": flat_contact_states,
                "weights": weights,
            }
            self.heel_toe_height_targets.append(target)
            self.task_errors[task] = []
            if self.contact_task_mode == "legacy_hold":
                self.tasks.append(task)
            summary["feet"][side] = {
                "simultaneous_contact_frames": int(
                    np.count_nonzero(flat_contact_states)
                ),
                "positive_weight_frames": int(np.count_nonzero(weights > 0.0)),
                "full_weight_frames": int(np.count_nonzero(weights >= 1.0 - 1.0e-9)),
            }
        self.heel_toe_height_summary = summary

    @staticmethod
    def _ramp_binary_weights(desired: np.ndarray, ramp_frames: int) -> np.ndarray:
        """把 0/1 接触意图变成逐帧线性权重，避免接触代价瞬时开关。

        这是有状态的因果进度限制器：进入和退出时接触进度每帧最多变化
        ``1 / ramp_frames``，再通过五次 smootherstep 映射为实际二次目标权重。
        这样首帧权重远小于线性坡道、两端一阶和二阶导数均为零，不会因为一帧
        检测抖动把高权重任务突然塞进或移出 QP；离散相位仍单独保留给落地投影
        和质量验收。
        """
        desired = np.asarray(desired, dtype=bool)
        if desired.ndim != 1:
            raise ValueError(f"desired contact weights must be 1D, got {desired.shape}")
        ramp_frames = max(1, int(ramp_frames))
        step = 1.0 / float(ramp_frames)
        progress = np.zeros(desired.shape, dtype=np.float64)
        previous_progress = 0.0
        for frame_idx, active in enumerate(desired):
            target = 1.0 if active else 0.0
            previous_progress += float(
                np.clip(target - previous_progress, -step, step)
            )
            progress[frame_idx] = previous_progress
        return progress**3 * (progress * (progress * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _true_intervals(states: np.ndarray) -> list[tuple[int, int]]:
        """把布尔状态转为半开支撑区间，供左右脚成对目标复用。"""
        states = np.asarray(states, dtype=bool)
        padded = np.r_[False, states, False]
        starts = np.flatnonzero(padded[1:] & ~padded[:-1])
        ends = np.flatnonzero(~padded[1:] & padded[:-1])
        return list(zip(starts.tolist(), ends.tolist()))

    @classmethod
    def _anticipatory_contact_weights(
        cls,
        desired: np.ndarray,
        contact_weights: np.ndarray,
        ramp_frames: int,
    ) -> np.ndarray:
        """生成接触前的竖直预约束权重，并与正式锁点任务平滑交接。

        每个支撑区间开始前 ``ramp_frames`` 帧，权重按五次 smootherstep 从 0
        增到 1；正式接触建立后使用 ``1 - contact_weight``，让仅 Z 方向的预约束
        与完整 XYZ 锁点的二次目标权重之和保持为 1，直到正式任务完全接管。这样
        腿部能在落地前逐步调整长度，且接触首帧不会突然插入高权重任务。
        """
        desired = np.asarray(desired, dtype=bool)
        contact_weights = np.asarray(contact_weights, dtype=np.float64)
        if desired.shape != contact_weights.shape or desired.ndim != 1:
            raise ValueError(
                "anticipatory contact inputs must be matching 1D arrays, got "
                f"{desired.shape}/{contact_weights.shape}"
            )
        ramp_frames = max(1, int(ramp_frames))
        weights = np.zeros(desired.shape, dtype=np.float64)
        for start, end in cls._true_intervals(desired):
            pre_start = max(0, start - ramp_frames)
            count = start - pre_start
            if count:
                progress = np.arange(1, count + 1, dtype=np.float64) / float(count)
                weights[pre_start:start] = progress**3 * (
                    progress * (progress * 6.0 - 15.0) + 10.0
                )
            weights[start:end] = np.maximum(
                weights[start:end], 1.0 - contact_weights[start:end]
            )
        return np.clip(weights, 0.0, 1.0)

    def _setup_paired_support_plan(self) -> None:
        """生成左右脚“平足/脚尖离地/摆动”成对支撑计划。

        脚跟检测为真时定义为平足期，脚跟和脚尖同时锁到同一地面高度；只有脚尖
        为真时才进入 toe-off，保持脚尖而释放脚跟。每个连续支撑区间使用源动作
        的平均脚尖 XY 与平均朝向，但使用机器人自身 heel-toe 距离重建脚跟目标，
        从几何上消除把 SMPL 脚掌倾角直接复制到 BUMI3 后造成的翘跟。
        """
        by_name = {item["contact_name"]: item for item in self.contact_targets}
        required = {
            "left_foot_end",
            "left_toe",
            "right_foot_end",
            "right_toe",
        }
        missing = sorted(required - set(by_name))
        if missing:
            raise ValueError(
                "paired_support requires heel/toe contact mappings for both feet; "
                f"missing={missing}"
            )

        ramp_frames = max(1, int(round(self.contact_weight_ramp_seconds * self.fps)))
        summary = {"ramp_frames": ramp_frames, "feet": {}}
        for side in ("left", "right"):
            heel = by_name[f"{side}_foot_end"]
            toe = by_name[f"{side}_toe"]
            heel_states = self.contact_seq[:, heel["contact_state_idx"]].astype(bool)
            toe_states = self.contact_seq[:, toe["contact_state_idx"]].astype(bool)
            # 2=平足，1=toe-off，0=摆动。脚跟一旦稳定接触，脚尖必须成对支撑。
            modes = np.where(heel_states, 2, np.where(toe_states, 1, 0)).astype(np.int8)
            support_states = modes > 0
            flat_states = modes == 2

            heel_source = np.asarray(
                self.keypoints_pos[
                    :, self.keypoint_name_to_idx[heel["source_keypoint_name"]], :
                ],
                dtype=np.float64,
            )
            toe_source = np.asarray(
                self.keypoints_pos[
                    :, self.keypoint_name_to_idx[toe["source_keypoint_name"]], :
                ],
                dtype=np.float64,
            )
            heel_targets = heel_source.copy()
            toe_targets = toe_source.copy()

            heel_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, heel["body_name"]
            )
            toe_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, toe["body_name"]
            )
            foot_body_id = int(self.model.body_parentid[heel_body_id])
            foot_body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, foot_body_id
            )
            flat_orientation_targets = np.broadcast_to(
                self.configuration.data.xquat[foot_body_id].copy(),
                (self.num_frames, 4),
            ).copy()
            initial_span_vector = (
                self.configuration.data.xpos[toe_body_id]
                - self.configuration.data.xpos[heel_body_id]
            )
            foot_span = float(np.linalg.norm(initial_span_vector[:2]))
            if foot_span <= 1.0e-6:
                foot_span = float(np.linalg.norm(initial_span_vector))
            if foot_span <= 1.0e-6:
                raise ValueError(f"{side} heel-toe marker span is zero")
            fallback_heading = initial_span_vector[:2]
            fallback_norm = float(np.linalg.norm(fallback_heading))
            fallback_heading = (
                fallback_heading / fallback_norm
                if fallback_norm > 1.0e-8
                else np.array([1.0, 0.0], dtype=np.float64)
            )

            support_intervals = self._true_intervals(support_states)
            for interval_idx, (start, end) in enumerate(support_intervals):
                if self.contact_position_aggregation == "first":
                    toe_xy = toe_source[start, :2]
                else:
                    toe_xy = np.mean(toe_source[start:end, :2], axis=0)
                source_headings = toe_source[start:end, :2] - heel_source[start:end, :2]
                norms = np.linalg.norm(source_headings, axis=1)
                valid = norms > 1.0e-8
                if np.any(valid):
                    heading = np.mean(
                        source_headings[valid] / norms[valid, None], axis=0
                    )
                    heading_norm = float(np.linalg.norm(heading))
                    heading = (
                        heading / heading_norm
                        if heading_norm > 1.0e-8
                        else fallback_heading
                    )
                else:
                    heading = fallback_heading
                toe_target = np.array(
                    [toe_xy[0], toe_xy[1], self.post_ik_ground_clearance],
                    dtype=np.float64,
                )
                heel_target = np.array(
                    [
                        toe_xy[0] - heading[0] * foot_span,
                        toe_xy[1] - heading[1] * foot_span,
                        self.post_ik_ground_clearance,
                    ],
                    dtype=np.float64,
                )
                toe_targets[start:end] = toe_target
                heel_targets[start:end] = heel_target
                flat_rotation = np.column_stack(
                    (
                        np.array([heading[0], heading[1], 0.0], dtype=np.float64),
                        np.array([-heading[1], heading[0], 0.0], dtype=np.float64),
                        np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    )
                )
                flat_quaternion_xyzw = Rot.from_matrix(flat_rotation).as_quat()
                flat_quaternion_wxyz = flat_quaternion_xyzw[[3, 0, 1, 2]]
                flat_orientation_targets[start:end] = flat_quaternion_wxyz
                # 离开支撑后权重仍按 smootherstep 渐出；这段时间必须继续持有上一
                # 锁点，不能在第一帧摆动时把目标瞬间换回源脚点。若下一支撑区间
                # 更早开始，则只持有到新区间边界，由接触检测的最短摆动时间保证
                # 权重已经充分衰减。
                next_start = (
                    support_intervals[interval_idx + 1][0]
                    if interval_idx + 1 < len(support_intervals)
                    else self.num_frames
                )
                hold_end = min(end + ramp_frames, next_start, self.num_frames)
                toe_targets[end:hold_end] = toe_target
                heel_targets[end:hold_end] = heel_target
                flat_orientation_targets[end:hold_end] = flat_quaternion_wxyz

            heel_weights = self._ramp_binary_weights(flat_states, ramp_frames)
            toe_weights = self._ramp_binary_weights(support_states, ramp_frames)
            heel_precontact_weights = self._anticipatory_contact_weights(
                flat_states, heel_weights, ramp_frames
            )
            toe_precontact_weights = self._anticipatory_contact_weights(
                support_states, toe_weights, ramp_frames
            )
            heel.update(
                paired_lock_positions=heel_targets,
                paired_weights=heel_weights,
                paired_ground_active=flat_states,
                paired_modes=modes,
            )
            toe.update(
                paired_lock_positions=toe_targets,
                paired_weights=toe_weights,
                paired_ground_active=support_states,
                paired_modes=modes,
            )
            for contact_target, weights in (
                (heel, heel_precontact_weights),
                (toe, toe_precontact_weights),
            ):
                precontact_positions = np.asarray(
                    contact_target["paired_lock_positions"], dtype=np.float64
                ).copy()
                precontact_positions[:, 2] = self.post_ik_ground_clearance
                vertical_task = mink.FrameTask(
                    frame_name=contact_target["body_name"],
                    frame_type="body",
                    position_cost=np.zeros(3, dtype=np.float64),
                    orientation_cost=0.0,
                    lm_damping=1.0,
                )
                self.paired_precontact_targets.append(
                    {
                        "side": side,
                        "body_name": contact_target["body_name"],
                        "task": vertical_task,
                        "base_position_cost": contact_target["base_position_cost"],
                        "positions": precontact_positions,
                        "weights": weights,
                    }
                )
            if self.paired_flat_orientation_cost > 0.0:
                orientation_task = mink.FrameTask(
                    frame_name=foot_body_name,
                    frame_type="body",
                    position_cost=0.0,
                    orientation_cost=self.paired_flat_orientation_cost,
                    lm_damping=1.0,
                )
                self.task_errors[orientation_task] = []
                self.paired_orientation_targets.append(
                    {
                        "side": side,
                        "body_name": foot_body_name,
                        "task": orientation_task,
                        "base_orientation_cost": self.paired_flat_orientation_cost,
                        "quaternions": flat_orientation_targets,
                        "weights": heel_weights,
                    }
                )
            summary["feet"][side] = {
                "flat_frames": int(np.count_nonzero(flat_states)),
                "toe_off_frames": int(np.count_nonzero(modes == 1)),
                "swing_frames": int(np.count_nonzero(modes == 0)),
                "support_intervals": len(self._true_intervals(support_states)),
                "heel_toe_span_m": foot_span,
                "heel_precontact_frames": int(
                    np.count_nonzero(heel_precontact_weights > 0.0)
                ),
                "toe_precontact_frames": int(
                    np.count_nonzero(toe_precontact_weights > 0.0)
                ),
            }
        self.paired_support_summary = summary

    def _parse_explicit_contact_map(self, value: dict) -> list[ContactMapping]:
        if not isinstance(value, dict):
            raise TypeError(f"contact_map must be a mapping, got {type(value).__name__}")
        mappings = []
        used_frames = set()
        for source_contact_name, entry in value.items():
            if not isinstance(entry, dict):
                raise TypeError(f"contact_map.{source_contact_name} must be a mapping")
            if not bool(entry.get("enabled", True)) or not bool(
                entry.get("lock_position", True)
            ):
                continue
            source_contact_name = str(source_contact_name)
            if source_contact_name not in self.contact_state_name_to_idx:
                raise ValueError(
                    f"contact_map source contact not found: {source_contact_name}; "
                    f"available={self.contact_names}"
                )
            frame_name = str(entry.get("frame_name", "")).strip()
            source_keypoint_name = str(entry.get("source_keypoint_name", "")).strip()
            frame_type = str(entry.get("frame_type", "body"))
            position_cost = float(entry.get("position_cost", self.contact_position_cost))
            if frame_type != "body":
                raise ValueError(
                    f"contact_map.{source_contact_name}.frame_type must be body, got {frame_type}"
                )
            if position_cost < 0.0 or not np.isfinite(position_cost):
                raise ValueError(
                    f"contact_map.{source_contact_name}.position_cost must be non-negative, "
                    f"got {position_cost}"
                )
            if source_keypoint_name not in self.keypoint_name_to_idx:
                raise ValueError(
                    f"contact_map source keypoint not found: {source_keypoint_name}; "
                    f"available={self.keypoint_names}"
                )
            if mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, frame_name
            ) < 0:
                raise ValueError(f"contact_map target body not found: {frame_name}")
            if frame_name in used_frames:
                raise ValueError(f"contact_map target body is mapped more than once: {frame_name}")
            used_frames.add(frame_name)
            mappings.append(
                ContactMapping(
                    source_contact_name=source_contact_name,
                    frame_name=frame_name,
                    source_keypoint_name=source_keypoint_name,
                    frame_type=frame_type,
                    position_cost=position_cost,
                )
            )
        return mappings

    def _build_legacy_contact_mappings(self) -> list[ContactMapping]:
        if not self.contact_body_names:
            return []
        if len(self.contact_body_names) != len(self.contact_names):
            if self.verbose:
                print(
                    "[contact target] skip: robot contact_links count "
                    f"{len(self.contact_body_names)} != keypoint contact_names count {len(self.contact_names)}"
                )
            return []

        mappings = []
        for body_name, contact_name in zip(self.contact_body_names, self.contact_names):
            contact_state_idx = self.contact_state_name_to_idx.get(contact_name)
            if contact_state_idx is None:
                continue

            source_keypoint_name = self._resolve_body_source_keypoint(body_name)
            if source_keypoint_name is None:
                if self.verbose:
                    print(f"[contact target] could not find a source keypoint for {body_name}")
                continue

            mappings.append(
                ContactMapping(
                    source_contact_name=contact_name,
                    frame_name=body_name,
                    source_keypoint_name=source_keypoint_name,
                    frame_type="body",
                    position_cost=self.contact_position_cost,
                )
            )
        return mappings

    def _resolve_body_source_keypoint(self, body_name):
        if body_name in self.keypoint_name_to_idx:
            return body_name
        return self.body_name_to_source_keypoint.get(body_name)

    def _ensure_contact_task(self, body_name, position_cost):
        existing_task = self.body_name_to_contact_task.get(body_name)
        if existing_task is not None:
            return existing_task

        task = mink.FrameTask(
            frame_name=body_name,
            frame_type="body",
            position_cost=position_cost,
            orientation_cost=0.0,
            lm_damping=1,
        )
        self.body_name_to_contact_task[body_name] = task
        self.task_errors[task] = []
        return task

    def _get_target_pose(self, frame_idx, keypoint_name):
        keypoint_idx = self.keypoint_name_to_idx[keypoint_name]
        target_pos = self.keypoints_pos[frame_idx, keypoint_idx]
        target_quat = self.keypoints_quat[frame_idx, keypoint_idx]
        return target_pos, target_quat

    def _get_body_pose(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body not found: {body_name}")
        return self.configuration.data.xpos[body_id], self.configuration.data.xquat[body_id]

    def _build_contact_lock_positions(self, contact_state_idx, source_keypoint_name):
        """为每个接触区间生成固定目标；first 可避免区间开始时跳向全段均值。"""
        lock_positions = np.zeros((self.num_frames, 3), dtype=np.float64)
        contact_states = self.contact_seq[:, contact_state_idx].astype(bool)
        if not np.any(contact_states):
            return lock_positions

        source_keypoint_idx = self.keypoint_name_to_idx[source_keypoint_name]
        source_positions = np.asarray(
            self.keypoints_pos[:, source_keypoint_idx, :], dtype=np.float64
        )

        frame_idx = 0
        while frame_idx < self.num_frames:
            if not contact_states[frame_idx]:
                frame_idx += 1
                continue

            interval_end = frame_idx + 1
            while interval_end < self.num_frames and contact_states[interval_end]:
                interval_end += 1

            if self.contact_position_aggregation == "first":
                interval_target = source_positions[frame_idx]
            else:
                interval_target = np.mean(
                    source_positions[frame_idx:interval_end], axis=0
                )
            lock_positions[frame_idx:interval_end] = interval_target
            if self.contact_position_aggregation == "mean_ramp":
                interval_length = interval_end - frame_idx
                transition_frames = max(
                    1,
                    int(round(self.contact_transition_window_seconds * self.fps)),
                )
                offsets = np.arange(interval_length, dtype=np.float64)
                if transition_frames == 1:
                    mean_weights = np.ones(interval_length, dtype=np.float64)
                else:
                    enter_weights = offsets / float(transition_frames - 1)
                    exit_weights = (interval_length - 1 - offsets) / float(
                        transition_frames - 1
                    )
                    mean_weights = np.clip(
                        np.minimum(enter_weights, exit_weights), 0.0, 1.0
                    )
                source_segment = source_positions[frame_idx:interval_end]
                lock_positions[frame_idx:interval_end] = (
                    (1.0 - mean_weights[:, None]) * source_segment
                    + mean_weights[:, None] * interval_target
                )
            frame_idx = interval_end

        return lock_positions

    def _get_contact_locked_position(self, frame_idx, contact_target):
        contact_state_idx = contact_target["contact_state_idx"]
        source_keypoint_name = contact_target["source_keypoint_name"]

        target_pos, _ = self._get_target_pose(frame_idx, source_keypoint_name)
        in_contact = bool(self.contact_seq[frame_idx, contact_state_idx])
        if in_contact:
            return contact_target["lock_positions"][frame_idx], True
        return target_pos, False
    
    def _apply_joints_limit_offset(self):
        """Add offsets to the lower/upper limits of matching joints per joints_limit_offset_degrees.

        The key is a joint-name substring (e.g. knee_joint / elbow_joint) and matches all joints
        in the model whose name contains that substring (e.g. left_knee_joint, right_knee_joint).
        The value is [lower_offset_deg, upper_offset_deg]: the lower offset is added to the lower
        bound and the upper offset to the upper bound (positive raises, negative lowers). A single
        scalar value is also supported (applied to the lower bound only).
        """
        if not self.joints_limit_offset_degrees:
            return

        # Collect all joint names in the model
        all_joint_names = []
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is not None:
                all_joint_names.append((joint_id, name))

        for key, offset_deg in self.joints_limit_offset_degrees.items():
            # Support [lower_offset, upper_offset] or a single scalar (lower bound only)
            if isinstance(offset_deg, (list, tuple)):
                if len(offset_deg) != 2:
                    raise ValueError(
                        f"joints_limit_offset_degrees['{key}'] must be two values [lower, upper], "
                        f"got {offset_deg}"
                    )
                lower_offset_rad = np.radians(float(offset_deg[0]))
                upper_offset_rad = np.radians(float(offset_deg[1]))
            else:
                lower_offset_rad = np.radians(float(offset_deg))
                upper_offset_rad = 0.0

            if lower_offset_rad == 0.0 and upper_offset_rad == 0.0:
                continue
            matched = [(jid, name) for jid, name in all_joint_names if key in name]
            if not matched:
                if self.verbose:
                    print(f"[joint limit] no matching joint found: {key}")
                continue
            for joint_id, joint_name in matched:
                lower, upper = self.model.jnt_range[joint_id]
                new_lower = lower + lower_offset_rad
                new_upper = upper + upper_offset_rad
                # Ensure the lower bound does not exceed the upper bound
                if new_lower > new_upper:
                    new_lower, new_upper = new_upper, new_lower
                self.model.jnt_range[joint_id, 0] = new_lower
                self.model.jnt_range[joint_id, 1] = new_upper
                if self.verbose:
                    print(
                        f"[joint limit] {joint_name} range: "
                        f"[{lower:.4f}, {upper:.4f}] -> [{new_lower:.4f}, {new_upper:.4f}] rad"
                    )
 
    def update_targets(self, frame_idx):
        # Record the keypoint targets (pos, quat_wxyz) mapped to tasks for the current frame, for visualization
        self.current_targets = []
        self.current_contact_points = []
        self.current_ground_body_ids = []
        if self.temporal_posture_task is not None:
            # 此时 configuration 仍是上一输出帧；整个当前帧的多次 IK 迭代都保持
            # 同一目标，从而在多个近似解之间优先选择连续的关节分支。
            self.temporal_posture_task.set_target_from_configuration(
                self.configuration
            )
        for keypoint_name, task in self.human_body_to_task.items():
            target_pos, target_quat = self._get_target_pose(frame_idx, keypoint_name)
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos))
            robot_frame, pos_weight, rot_weight = self.ik_match_table[keypoint_name]
            self.current_targets.append(
                {
                    "pos": np.asarray(target_pos, dtype=np.float64),
                    "quat": np.asarray(target_quat, dtype=np.float64),
                    "robot_frame": robot_frame,
                    "pos_weight": float(pos_weight),
                    "rot_weight": float(rot_weight),
                }
            )

        for height_target in self.heel_toe_height_targets:
            weight = float(height_target["weights"][frame_idx])
            # Mink 的 cost 会平方后进入 QP，因此乘 sqrt(weight) 才是线性渐变。
            height_target["task"].set_height_cost(
                height_target["base_cost"] * np.sqrt(weight)
            )

        if self.contact_task_mode == "paired_support":
            # FrameTask 的二次目标按 cost^2 进入 QP，因此使用 sqrt(weight) 才让
            # 实际目标权重线性渐入/渐出；任务对象始终留在列表中，QP 维度不跳变。
            for contact_target in self.contact_targets:
                target_pos = np.asarray(
                    contact_target["paired_lock_positions"][frame_idx],
                    dtype=np.float64,
                )
                weight = float(contact_target["paired_weights"][frame_idx])
                contact_target["task"].set_position_cost(
                    contact_target["base_position_cost"] * np.sqrt(weight)
                )
                _, target_quat = self._get_target_pose(
                    frame_idx, contact_target["source_keypoint_name"]
                )
                contact_target["task"].set_target(
                    mink.SE3.from_rotation_and_translation(
                        mink.SO3(target_quat), target_pos
                    )
                )
                if weight > 0.0:
                    self.current_contact_points.append(target_pos)
                if (
                    bool(contact_target["paired_ground_active"][frame_idx])
                    and (frame_idx == 0 or weight >= 1.0 - 1.0e-9)
                ):
                    body_id = mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        contact_target["body_name"],
                    )
                    self.current_ground_body_ids.append(body_id)
            for orientation_target in self.paired_orientation_targets:
                weight = float(orientation_target["weights"][frame_idx])
                orientation_target["task"].set_orientation_cost(
                    orientation_target["base_orientation_cost"] * np.sqrt(weight)
                )
                body_position, _ = self._get_body_pose(
                    orientation_target["body_name"]
                )
                target_quaternion = orientation_target["quaternions"][frame_idx]
                orientation_target["task"].set_target(
                    mink.SE3.from_rotation_and_translation(
                        mink.SO3(target_quaternion), body_position
                    )
                )
            for precontact_target in self.paired_precontact_targets:
                weight = float(precontact_target["weights"][frame_idx])
                precontact_target["task"].set_position_cost(
                    np.array(
                        [
                            0.0,
                            0.0,
                            precontact_target["base_position_cost"]
                            * np.sqrt(weight),
                        ],
                        dtype=np.float64,
                    )
                )
                target_position = np.asarray(
                    precontact_target["positions"][frame_idx], dtype=np.float64
                )
                precontact_target["task"].set_target(
                    mink.SE3.from_rotation_and_translation(
                        mink.SO3.identity(), target_position
                    )
                )
            return self.base_tasks + [
                item["task"] for item in self.contact_targets
            ] + [item["task"] for item in self.paired_orientation_targets] + [
                item["task"] for item in self.paired_precontact_targets
            ] + [item["task"] for item in self.heel_toe_height_targets]

        active_contact_tasks = []
        for contact_target in self.contact_targets:
            source_keypoint_name = contact_target["source_keypoint_name"]
            target_pos, is_active = self._get_contact_locked_position(frame_idx, contact_target)
            if is_active:
                _, target_quat = self._get_target_pose(frame_idx, source_keypoint_name)
                contact_target["task"].set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos)
                )
                active_contact_tasks.append(contact_target["task"])
                self.current_contact_points.append(np.asarray(target_pos, dtype=np.float64))
            else:
                if self.contact_task_mode == "legacy_hold":
                    target_pos, target_quat = self._get_body_pose(contact_target["body_name"])
                    contact_target["task"].set_target(
                        mink.SE3.from_rotation_and_translation(
                            mink.SO3(target_quat), target_pos
                        )
                    )
        if self.contact_task_mode == "active_only":
            return self.base_tasks + active_contact_tasks + [
                item["task"] for item in self.heel_toe_height_targets
            ]
        return self.tasks
    
    def error(self, tasks):
        residuals = []
        for task in tasks:
            error = task.compute_error(self.configuration)
            if self.ik_error_metric == "solver_weighted" and hasattr(task, "cost"):
                error = np.asarray(task.cost) * error
            residuals.append(error)
        return np.linalg.norm(
            np.concatenate(residuals)
        )
    
    def _draw_pose(self, scene, pos, quat_wxyz, point_rgba, axis_alpha=1.0,
                   point_radius=0.035, axis_radius=0.005, axis_length=0.08):
        """Draw a pose in the scene: one small sphere + red/green/blue (XYZ) axes. Returns whether it succeeded."""
        max_geoms = len(scene.geoms)
        if scene.ngeom + 4 > max_geoms:
            return False

        pos = np.asarray(pos, dtype=np.float64)
        identity_mat = np.eye(3, dtype=np.float64).reshape(-1)
        axis_colors = (
            np.array([1.0, 0.0, 0.0, axis_alpha], dtype=np.float32),  # X red
            np.array([0.0, 1.0, 0.0, axis_alpha], dtype=np.float32),  # Y green
            np.array([0.0, 0.0, 1.0, axis_alpha], dtype=np.float32),  # Z blue
        )

        # Position sphere
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([point_radius, point_radius, point_radius], dtype=np.float64),
            pos,
            identity_mat,
            np.asarray(point_rgba, dtype=np.float32),
        )
        scene.ngeom += 1

        # Derive the rotation matrix from the quaternion (wxyz); its columns are the axis directions
        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, np.asarray(quat_wxyz, dtype=np.float64))
        rot_mat = rot_mat.reshape(3, 3)
        # A cylinder defaults along its local z axis, so rotate each axis to its target direction
        base_to_axis = (
            Rot.from_euler("y", 90, degrees=True).as_matrix(),   # z->x
            Rot.from_euler("x", -90, degrees=True).as_matrix(),  # z->y
            np.eye(3),                                           # z->z
        )
        for axis_idx in range(3):
            axis_world_rot = rot_mat @ base_to_axis[axis_idx]
            axis_dir = rot_mat[:, axis_idx]
            center = pos + axis_dir * (0.5 * axis_length)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([axis_radius, 0.5 * axis_length, 0.0], dtype=np.float64),
                center,
                axis_world_rot.reshape(-1),
                axis_colors[axis_idx],
            )
            scene.ngeom += 1
        return True

    def _draw_point(self, scene, pos, point_rgba, point_radius=0.04):
        """Draw only a single spherical point in the scene."""
        max_geoms = len(scene.geoms)
        if scene.ngeom + 1 > max_geoms:
            return False

        pos = np.asarray(pos, dtype=np.float64)
        identity_mat = np.eye(3, dtype=np.float64).reshape(-1)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([point_radius, point_radius, point_radius], dtype=np.float64),
            pos,
            identity_mat,
            np.asarray(point_rgba, dtype=np.float32),
        )
        scene.ngeom += 1
        return True

    def draw_target_keypoints(self, viewer, point_radius=0.035, axis_radius=0.005, axis_length=0.08, targets=None):
        """Draw the positions and axes of the target keypoints and the corresponding robot bodies.

        - Target keypoint: yellow sphere + full-brightness axes.
        - Robot body: cyan sphere + semi-transparent axes.
        """
        if targets is None:
            targets = getattr(self, "current_targets", None)
        scene = viewer.user_scn
        scene.ngeom = 0

        target_rgba = np.array([1.0, 1.0, 0.0, 0.9], dtype=np.float32)   # yellow
        robot_rgba = np.array([0.0, 1.0, 1.0, 0.9], dtype=np.float32)    # cyan
        contact_rgba = np.array([1.0, 0.0, 0.0, 0.95], dtype=np.float32) # red

        # 1) Target keypoints
        if targets:
            for target in targets:
                if isinstance(target, dict):
                    target_pos = target["pos"]
                    target_quat_wxyz = target["quat"]
                    robot_frame = target.get("robot_frame")
                    pos_weight = target.get("pos_weight", 1.0)
                    rot_weight = target.get("rot_weight", 1.0)
                else:
                    # Backward compatibility with the old (pos, quat) tuple format
                    target_pos, target_quat_wxyz = target
                    robot_frame = None
                    pos_weight = 1.0
                    rot_weight = 1.0

                # Zero position weight: draw the sphere at the corresponding body's position (overlapping)
                if pos_weight == 0 and robot_frame is not None:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_frame)
                    if body_id >= 0:
                        target_pos = self.configuration.data.xpos[body_id]

                if rot_weight == 0:
                    # Zero orientation weight: show only the sphere, no axes
                    if not self._draw_point(
                        scene, target_pos, target_rgba,
                        point_radius=point_radius,
                    ):
                        break
                else:
                    if not self._draw_pose(
                        scene, target_pos, target_quat_wxyz, target_rgba,
                        axis_alpha=1.0,
                        point_radius=point_radius, axis_radius=axis_radius, axis_length=axis_length,
                    ):
                        break

        # 2) Current position and orientation of the corresponding robot bodies
        data = self.configuration.data
        for robot_frame in self.robot_frame_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_frame)
            if body_id < 0:
                continue
            body_pos = data.xpos[body_id]
            body_quat = data.xquat[body_id]  # wxyz
            if not self._draw_pose(
                scene, body_pos, body_quat, robot_rgba,
                axis_alpha=0.5,
                point_radius=point_radius, axis_radius=axis_radius, axis_length=axis_length,
            ):
                break

        # 3) Currently active contact points, drawn as red spheres
        for contact_pos in self.current_contact_points:
            if not self._draw_point(
                scene,
                contact_pos,
                contact_rgba,
                point_radius=max(point_radius * 1.5, 0.02),
            ):
                break
    
    def retarget(self):
        global _PAUSED
        viewer = None
        _PAUSED = False

        if self.render_debug:
            viewer = mujoco.viewer.launch_passive(
                self.model, self.configuration.data, key_callback=key_callback
            )
            print("Controls: Space play/pause")

        try:
            for frame_idx in tqdm(range(self.num_frames), desc="Retargeting", unit="frame"):
                start_time = time.time()
                previous_output_qpos = self.configuration.data.qpos.copy()
                solve_tasks = self.update_targets(frame_idx)

                curr_error = self.error(solve_tasks)
                dt = self.configuration.model.opt.timestep

                vel = mink.solve_ik(
                    self.configuration, solve_tasks, dt, self.solver, self.damping, limits=self.ik_limits
                )
                self.configuration.integrate_inplace(vel, dt)
                next_error = self.error(solve_tasks)
                num_iter = 1
                while num_iter < self.max_iter and (
                    (
                        frame_idx == 0
                        and num_iter < self.initial_settle_iterations
                    )
                    or curr_error - next_error
                    > self.ik_error_improvement_threshold
                ):
                    curr_error = next_error
                    dt = self.configuration.model.opt.timestep
                    vel = mink.solve_ik(
                        self.configuration, solve_tasks, dt, self.solver, self.damping, limits=self.ik_limits
                    )
                    self.configuration.integrate_inplace(vel, dt)
                    next_error = self.error(solve_tasks)
                    num_iter += 1
                clip_count, raw_max_joint_velocity = (
                    self._apply_output_joint_velocity_limit(previous_output_qpos)
                )
                ground_correction = self._apply_post_ik_ground_clearance()
                (
                    floor_barrier_adjustment_count,
                    floor_barrier_max_adjustment,
                ) = self._apply_foot_floor_barrier(
                    frame_idx, previous_output_qpos
                )
                next_error = self.error(solve_tasks)
                self.frame_ground_corrections.append(float(ground_correction))
                self.frame_floor_barrier_adjustment_counts.append(
                    int(floor_barrier_adjustment_count)
                )
                self.frame_floor_barrier_max_adjustments.append(
                    float(floor_barrier_max_adjustment)
                )
                self.frame_output_velocity_clip_counts.append(int(clip_count))
                self.frame_raw_max_joint_velocities.append(
                    float(raw_max_joint_velocity)
                )
                self.frame_final_errors.append(float(next_error))
                self.frame_iteration_counts.append(int(num_iter))
                curr_pos = self.configuration.data.qpos.copy()
                self.result_pos.append(curr_pos)

                if viewer is not None:
                    if not viewer.is_running():
                        break
                    mujoco.mj_forward(self.model, self.configuration.data)
                    self.draw_target_keypoints(viewer)
                    viewer.sync()
                    # Stay on the current frame while paused, until unpaused or the window is closed
                    while _PAUSED and viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                    if not viewer.is_running():
                        break
                    end_time = time.time()
                    elapsed = end_time - start_time
                    time.sleep(max(0, self.time_step - elapsed))
                    # time.sleep(0.02)  # reduce CPU usage

            self._postprocess_result_trajectory()
        finally:
            if viewer is not None:
                viewer.close()
    
    def save_results_as_csv(self, output_path):
        if len(self.result_pos) == 0:
            raise ValueError("No retarget results to save. Run retarget() first.")

        # [num_frame, num_joint], the first 7 dims are posXYZ + quat(wxyz)
        result = np.asarray(self.result_pos, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] < 7:
            raise ValueError(
                f"result_pos shape must be [num_frame, num_joint>=7], got {result.shape}"
            )

        # Reorder the first-7-column quaternion from wxyz (cols 3..6) to xyzw
        result_xyzw = result.copy()
        result_xyzw[:, 3:7] = result[:, [4, 5, 6, 3]]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(result_xyzw.tolist())

        if self.verbose:
            print(f"Saved retarget results to: {output_path} (shape={result_xyzw.shape})")

    def save_metadata(self, output_path):
        """Save the qpos, contact, and IK contract beside the legacy CSV."""
        joint_names = []
        joint_qpos_addresses = {}
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                joint_names.append(name)
                joint_qpos_addresses[name] = int(self.model.jnt_qposadr[joint_id])

        contact_statistics = {}
        for target in self.contact_targets:
            states = self.contact_seq[:, target["contact_state_idx"]].astype(bool)
            starts = states & ~np.r_[False, states[:-1]]
            contact_statistics[target["contact_name"]] = {
                "active_frames": int(states.sum()),
                "interval_count": int(starts.sum()),
            }
        errors = np.asarray(self.frame_final_errors, dtype=np.float64)
        iterations = np.asarray(self.frame_iteration_counts, dtype=np.int64)
        ground_corrections = np.asarray(
            self.frame_ground_corrections, dtype=np.float64
        )
        velocity_clip_counts = np.asarray(
            self.frame_output_velocity_clip_counts, dtype=np.int64
        )
        raw_max_joint_velocities = np.asarray(
            self.frame_raw_max_joint_velocities, dtype=np.float64
        )
        floor_barrier_adjustment_counts = np.asarray(
            self.frame_floor_barrier_adjustment_counts, dtype=np.int64
        )
        floor_barrier_max_adjustments = np.asarray(
            self.frame_floor_barrier_max_adjustments, dtype=np.float64
        )
        ik_statistics = {
            "solver": self.solver,
            "damping": float(self.damping),
            "max_iterations": int(self.max_iter),
            "initial_settle_iterations": int(self.initial_settle_iterations),
            "improvement_threshold": float(self.ik_error_improvement_threshold),
            "error_metric": self.ik_error_metric,
            "final_error_per_frame": errors.tolist(),
            "final_error_mean": float(errors.mean()) if errors.size else None,
            "final_error_max": float(errors.max()) if errors.size else None,
            "iterations_per_frame": iterations.tolist(),
            "iterations_mean": float(iterations.mean()) if iterations.size else None,
            "iterations_max": int(iterations.max()) if iterations.size else None,
            "post_ik_ground_bodies": self.post_ik_ground_bodies,
            "post_ik_ground_clearance_m": float(self.post_ik_ground_clearance),
            "post_ik_ground_mode": self.post_ik_ground_mode,
            "post_ik_ground_correction_mean_m": (
                float(ground_corrections.mean()) if ground_corrections.size else 0.0
            ),
            "post_ik_ground_correction_min_m": (
                float(ground_corrections.min()) if ground_corrections.size else 0.0
            ),
            "post_ik_ground_correction_max_m": (
                float(ground_corrections.max()) if ground_corrections.size else 0.0
            ),
            "post_ik_ground_correction_max_abs_m": (
                float(np.max(np.abs(ground_corrections)))
                if ground_corrections.size
                else 0.0
            ),
            "max_output_joint_velocity_rad_s": float(
                self.max_output_joint_velocity_rad_s
            ),
            "max_output_joint_acceleration_rad_s2": float(
                self.max_output_joint_acceleration_rad_s2
            ),
            "max_output_joint_jerk_rad_s3": float(
                self.max_output_joint_jerk_rad_s3
            ),
            "output_velocity_clipped_frame_count": int(
                np.count_nonzero(velocity_clip_counts)
            ),
            "output_velocity_clipped_joint_sample_count": int(
                velocity_clip_counts.sum()
            ),
            "raw_max_output_joint_velocity_rad_s": (
                float(raw_max_joint_velocities.max())
                if raw_max_joint_velocities.size
                else 0.0
            ),
            "postprocess": self.postprocess_statistics,
            "foot_floor_barrier_adjusted_frame_count": int(
                np.count_nonzero(floor_barrier_adjustment_counts)
            ),
            "foot_floor_barrier_adjusted_foot_count": int(
                floor_barrier_adjustment_counts.sum()
            ),
            "foot_floor_barrier_max_ankle_adjustment_rad": (
                float(floor_barrier_max_adjustments.max())
                if floor_barrier_max_adjustments.size
                else 0.0
            ),
        }

        def digest(path_value):
            path = Path(path_value).resolve()
            hasher = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()

        payload = {
            "robot": self.robot_name,
            "source_motion": self.keypoints_metadata.get(
                "source_motion_file", str(Path(self.keypoint_path).resolve())
            ),
            "keypoints_path": str(Path(self.keypoint_path).resolve()),
            "fps": float(self.fps),
            "num_frames": int(len(self.result_pos)),
            "qpos_size": int(self.model.nq),
            "root_quaternion_order_in_csv": "xyzw",
            "mujoco_joint_names": joint_names,
            "mujoco_joint_qpos_addresses": joint_qpos_addresses,
            "isaac_joint_names": self.isaac_joint_names,
            "isaac_body_names": self.isaac_body_names,
            "body_aliases": self.body_aliases,
            "robot_xml_sha256": digest(self.xml_file),
            "config_sha256": digest(self.config_path) if self.config_path else None,
            "contact_task_mode": self.contact_task_mode,
            "contact_position_aggregation": self.contact_position_aggregation,
            "contact_transition_window_seconds": self.contact_transition_window_seconds,
            "contact_weight_ramp_seconds": self.contact_weight_ramp_seconds,
            "paired_flat_orientation_cost": self.paired_flat_orientation_cost,
            "paired_support_summary": self.paired_support_summary,
            "heel_toe_height_difference_cost": (
                self.heel_toe_height_difference_cost
            ),
            "heel_toe_height_difference_ramp_seconds": (
                self.heel_toe_height_difference_ramp_seconds
            ),
            "heel_toe_height_difference_summary": self.heel_toe_height_summary,
            "postprocess_joint_gaussian_sigma_frames": float(
                self.postprocess_joint_gaussian_sigma_frames
            ),
            "postprocess_support_projection": self.postprocess_support_projection,
            "postprocess_min_active_support_height_m": float(
                self.postprocess_min_active_support_height_m
            ),
            "postprocess_max_stable_support_height_above_clearance_m": float(
                self.postprocess_max_stable_support_height_above_clearance_m
            ),
            "temporal_posture_cost": self.temporal_posture_cost,
            "contact_statistics": contact_statistics,
            "ik_statistics": ik_statistics,
        }
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if self.verbose:
            print(f"Saved retarget metadata to: {output}")



if  __name__ == "__main__":
    args = parse_args()
    workspace_root = os.path.abspath(os.path.join(SCRIPT_DIR,"..", "robot", ".."))
    config_path = os.path.expanduser(args.config)
    if not os.path.isabs(config_path):
        config_path = os.path.join(workspace_root, config_path)

    print("mujoco version: ", mujoco.__version__)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    ik_match_table = config.get("ik_match_table", {})
    robot_xml_path = config.get("robot_xml_path", "")
    keypoints_path = config.get("keypoints_path", "")
    keypoints_idx = config.get("keypoints_idx","")

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(workspace_root, output_dir)
    if args.keypoints_name:
        keypoints_path = os.path.join(
            output_dir,
            "keypoints",
            config_name,
            f"{args.keypoints_name}_keypoints.pkl",
        )

    if robot_xml_path and not os.path.isabs(robot_xml_path):
        robot_xml_path = os.path.join(workspace_root, robot_xml_path)
    if keypoints_path and not os.path.isabs(keypoints_path):
        keypoints_path = os.path.join(workspace_root, keypoints_path)

    verbose_mode = config.get("verbose", False)
    config_render_debug = bool(config.get("render_debug", False))
    render_debug = config_render_debug if args.render_debug is None else bool(args.render_debug)
    joints_limit_offset_degrees = config.get("joints_limit_offset_degrees", {})
    contact_body_names = config.get("contact_links")
    if contact_body_names is None:
        contact_body_names = {
            field_name: config.get(field_name, [])
            for field_name in RobotRetarget.LEGACY_CONTACT_CONFIG_TO_NAMES
        }
    contact_position_cost = config.get("contact_pos_fixed_factor", 10.0)
    
    robot_retarget = RobotRetarget(
        model_path=robot_xml_path,
        keypoint_path=keypoints_path,
        ik_match_table=ik_match_table,
        solver=str(config.get("ik_solver", "daqp")),
        verbose=verbose_mode,
        damping=float(config.get("ik_damping", 1.0)),
        render_debug=render_debug,
        joints_limit_offset_degrees=joints_limit_offset_degrees,
        contact_body_names=contact_body_names,
        contact_position_cost=contact_position_cost,
        contact_map=config.get("contact_map"),
        contact_task_mode=str(config.get("contact_task_mode", "legacy_hold")),
        contact_position_aggregation=str(
            config.get("contact_position_aggregation", "mean")
        ),
        contact_transition_window_seconds=float(
            config.get("contact_transition_window_seconds", 0.0)
        ),
        contact_weight_ramp_seconds=float(
            config.get("contact_weight_ramp_seconds", 0.0)
        ),
        paired_flat_orientation_cost=float(
            config.get("paired_flat_orientation_cost", 0.0)
        ),
        heel_toe_height_difference_cost=float(
            config.get("heel_toe_height_difference_cost", 0.0)
        ),
        heel_toe_height_difference_ramp_seconds=float(
            config.get("heel_toe_height_difference_ramp_seconds", 0.0)
        ),
        initial_root_pose=config.get("initial_root_pose"),
        initial_joint_positions=config.get("initial_joint_positions"),
        max_ik_iterations=int(config.get("max_ik_iterations", 50)),
        initial_settle_iterations=int(config.get("initial_settle_iterations", 0)),
        ik_error_improvement_threshold=float(
            config.get("ik_error_improvement_threshold", 0.001)
        ),
        ik_error_metric=str(config.get("ik_error_metric", "legacy_raw")),
        robot_name=config_name,
        config_path=config_path,
        isaac_joint_names=config.get("isaac_joint_names", []),
        isaac_body_names=config.get("isaac_body_names", []),
        body_aliases=config.get("mjcf_to_isaac_body_name", {}),
        post_ik_ground_bodies=config.get("post_ik_ground_bodies", []),
        post_ik_ground_clearance=float(config.get("post_ik_ground_clearance", 0.0)),
        post_ik_ground_mode=str(config.get("post_ik_ground_mode", "lift_only")),
        post_ik_foot_floor_barrier=config.get("post_ik_foot_floor_barrier"),
        temporal_posture_cost=float(config.get("temporal_posture_cost", 0.0)),
        max_output_joint_velocity_rad_s=float(
            config.get("max_output_joint_velocity_rad_s", 0.0)
        ),
        max_output_joint_acceleration_rad_s2=float(
            config.get("max_output_joint_acceleration_rad_s2", 0.0)
        ),
        max_output_joint_jerk_rad_s3=float(
            config.get("max_output_joint_jerk_rad_s3", 0.0)
        ),
        postprocess_joint_gaussian_sigma_frames=float(
            config.get("postprocess_joint_gaussian_sigma_frames", 0.0)
        ),
        postprocess_support_projection=bool(
            config.get("postprocess_support_projection", False)
        ),
        postprocess_min_active_support_height_m=float(
            config.get("postprocess_min_active_support_height_m", -0.009)
        ),
        postprocess_max_stable_support_height_above_clearance_m=float(
            config.get(
                "postprocess_max_stable_support_height_above_clearance_m", 0.018
            )
        ),
    )

    robot_retarget.retarget()

    keypoint_stem = os.path.splitext(os.path.basename(keypoints_path))[0]
    if keypoint_stem.endswith("_keypoints"):
        keypoint_stem = keypoint_stem[: -len("_keypoints")]
    output_csv = os.path.join(
        output_dir,
        "robot_motion",
        f"{keypoint_stem}_{config_name}.csv",
    )
    robot_retarget.save_results_as_csv(output_csv)
    robot_retarget.save_metadata(os.path.splitext(output_csv)[0] + ".meta.json")
