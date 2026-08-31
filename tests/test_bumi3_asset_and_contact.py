"""BUMI3 派生资产和单阶段 IK 接触映射契约测试。

本文件直接加载仓库交付的 ``bumi3_retarget.xml``，核对 21 个目标关节、9 个
无质量无碰撞 marker、唯一 ground、右臂 roll 限位和左右脚 marker 几何。
随后用临时 keypoint PKL 实例化通用 ``RobotRetarget``：显式 map 只映射一个
contact，以证明它不再依赖 contact 列表等长或顺序；检查 ``active_only`` 的
兼容行为，并验证 BUMI3 新增的左右脚成对平足、toe-off 与权重渐变状态机。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from bumi3_common import BUMI3_JOINT_NAMES, BUMI3_MARKER_BODY_NAMES
from robot_retarget import RobotRetarget


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config/robot/bumi3.yaml"
XML_PATH = REPOSITORY_ROOT / "asset/robot/bumi3/mjcf/bumi3_retarget.xml"


def test_bumi3_production_config_uses_scoped_dynamic_foot_policy() -> None:
    """生产配置保留 G1 IK 基线，并把动态高度严格限制在四个足点。"""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    g1_config = yaml.safe_load(
        (REPOSITORY_ROOT / "config/robot/g1.yaml").read_text(encoding="utf-8")
    )
    assert config["contact_task_mode"] == "legacy_hold"
    assert config["contact_position_aggregation"] == "mean"
    assert config["contact_pos_fixed_factor"] == pytest.approx(15.0)
    assert "contact_map" not in config
    assert len(config["contact_links"]) == 6
    assert config["ground_reference_contacts"] == [
        "left_foot_end",
        "left_toe",
        "right_foot_end",
        "right_toe",
    ]
    assert config["contact_height_dynamic_offset_enabled"] is True
    assert config["contact_height_relative_to_sequence_floor"] is True
    assert config["contact_height_floor_method"] == "stable_support_dense_median"
    assert config["contact_height_floor_fit_speed_threshold_mps"] == pytest.approx(0.2)
    assert config["contact_height_floor_fit_inlier_tolerance_m"] == pytest.approx(0.04)
    assert config["contact_height_floor_fit_min_samples"] == 8
    assert "contact_height_floor_percentile" not in config
    assert config["heel_toe_height_difference_cost"] == pytest.approx(15.0)
    assert config["heel_toe_height_difference_ramp_seconds"] == pytest.approx(0.10)
    assert config["contact_hysteresis"]["enabled"] is False
    assert config["require_ground_contact_detection"] is False
    assert config["post_ik_ground_bodies"] == []
    assert config["postprocess_support_projection"] is False
    assert config["paired_flat_orientation_cost"] == 0.0
    assert config["joint_limit_occupancy_validation"] == "warning"
    assert config["ik_error_metric"] == "legacy_raw"
    assert config["initial_settle_iterations"] == 100
    assert config["temporal_posture_cost"] == 0.0
    assert config["max_output_joint_velocity_rad_s"] == 0.0
    assert config["max_output_joint_acceleration_rad_s2"] == 0.0
    assert config["max_output_joint_jerk_rad_s3"] == 0.0
    assert config["output"]["target_fps"] == pytest.approx(30.0)
    for name in config["ik_match_table"]:
        assert config["ik_match_table"][name][1:] == g1_config["ik_match_table"][name][1:]


def test_bumi3_asset_contract() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    actual_joints = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)
        for idx in range(model.njnt)
    }
    assert set(BUMI3_JOINT_NAMES).issubset(actual_joints)
    assert len(BUMI3_JOINT_NAMES) == 21
    spec = mujoco.MjSpec.from_file(str(XML_PATH))
    assert sum(geom.name == "ground" for geom in spec.geoms) == 1
    right_roll = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_arm_roll_joint")
    np.testing.assert_allclose(model.jnt_range[right_roll], [-1.94, 0.14])
    for name in BUMI3_MARKER_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_id >= 0
        assert model.body_mass[body_id] == 0.0
        start = int(model.body_geomadr[body_id])
        count = int(model.body_geomnum[body_id])
        for geom_id in range(start, start + count):
            assert model.geom_contype[geom_id] == 0
            assert model.geom_conaffinity[geom_id] == 0
            if name in {
                "left_foot_end_link",
                "left_toe_link",
                "right_foot_end_link",
                "right_toe_link",
            }:
                np.testing.assert_allclose(model.geom_rgba[geom_id], [1.0, 0.0, 0.0, 1.0])
            else:
                assert model.geom_rgba[geom_id, 3] == 0.0

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    marker = {
        name: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()
        for name in (
            "left_foot_end_link",
            "left_toe_link",
            "right_foot_end_link",
            "right_toe_link",
        )
    }
    assert marker["left_toe_link"][0] > marker["left_foot_end_link"][0]
    assert marker["right_toe_link"][0] > marker["right_foot_end_link"][0]
    np.testing.assert_allclose(
        marker["left_toe_link"][0] - marker["left_foot_end_link"][0],
        marker["right_toe_link"][0] - marker["right_foot_end_link"][0],
        rtol=0.01,
    )

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    skeleton = yaml.safe_load(
        (REPOSITORY_ROOT / "config/skeleton/skeleton.yaml").read_text(encoding="utf-8")
    )
    assert set(config["robot_links"]) == set(skeleton["skeleton_links"])


def write_keypoints(path: Path, contacts: list[str], states: np.ndarray) -> None:
    """写入包含 BUMI3 IK 源关键点和指定接触状态的最小 PKL。"""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    keypoint_names = list(config["ik_match_table"])
    for name in (
        "left_foot_end_link",
        "left_toe_link",
        "right_foot_end_link",
        "right_toe_link",
    ):
        if name not in keypoint_names:
            keypoint_names.append(name)
    frame_count = states.shape[0]
    payload = {
        "keypoint_names": keypoint_names,
        "positions": np.zeros((frame_count, len(keypoint_names), 3), dtype=np.float32),
        "quaternions": np.broadcast_to(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (frame_count, len(keypoint_names), 4),
        ).copy(),
        "fps": 50.0,
        "contact_names": contacts,
        "contact_states": states,
    }
    with path.open("wb") as file:
        pickle.dump(payload, file)


def test_explicit_contact_map_and_active_only(tmp_path: Path) -> None:
    keypoints_path = tmp_path / "keypoints.pkl"
    write_keypoints(
        keypoints_path,
        ["right_toe", "left_foot_end"],
        np.asarray([[False, True], [False, False]], dtype=bool),
    )
    retarget = RobotRetarget(
        model_path=str(XML_PATH),
        keypoint_path=str(keypoints_path),
        ik_match_table={"hips_mean": ["hips_sphere", 1.0, 0.0]},
        contact_body_names=[],
        contact_map={
            "left_foot_end": {
                "frame_name": "left_foot_end_link",
                "source_keypoint_name": "left_foot_end_link",
                "frame_type": "body",
                "position_cost": 10.0,
            }
        },
        contact_task_mode="active_only",
    )
    assert len(retarget.contact_targets) == 1
    active_tasks = retarget.update_targets(0)
    swing_tasks = retarget.update_targets(1)
    assert len(active_tasks) == len(retarget.base_tasks) + 1
    assert len(swing_tasks) == len(retarget.base_tasks)


def test_contact_mean_ramp_matches_source_at_both_boundaries(tmp_path: Path) -> None:
    keypoints_path = tmp_path / "keypoints.pkl"
    write_keypoints(
        keypoints_path,
        ["left_foot_end"],
        np.ones((5, 1), dtype=bool),
    )
    retarget = RobotRetarget(
        model_path=str(XML_PATH),
        keypoint_path=str(keypoints_path),
        ik_match_table={"hips_mean": ["hips_sphere", 1.0, 0.0]},
        contact_map={
            "left_foot_end": {
                "frame_name": "left_foot_end_link",
                "source_keypoint_name": "left_foot_end_link",
            }
        },
        contact_task_mode="active_only",
        contact_position_aggregation="mean_ramp",
        contact_transition_window_seconds=0.04,
    )
    source_idx = retarget.keypoint_name_to_idx["left_foot_end_link"]
    retarget.keypoints_pos[:, source_idx, 0] = np.arange(5, dtype=np.float64)
    lock_positions = retarget._build_contact_lock_positions(
        contact_state_idx=0,
        source_keypoint_name="left_foot_end_link",
    )
    assert np.allclose(lock_positions[:, 0], [0.0, 2.0, 2.0, 2.0, 4.0])


def test_heel_toe_height_soft_task_ramps_and_cancels_root_z(tmp_path: Path) -> None:
    """等高软任务仅在同脚 heel/toe 同时接触时渐变，且不直接作用 Root-Z。"""
    keypoints_path = tmp_path / "keypoints.pkl"
    contacts = ["left_foot_end", "left_toe", "right_foot_end", "right_toe"]
    states = np.asarray(
        [
            [False, False, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    write_keypoints(keypoints_path, contacts, states)
    contact_map = {
        name: {
            "frame_name": f"{name}_link",
            "source_keypoint_name": f"{name}_link",
            "position_cost": 15.0,
        }
        for name in contacts
    }
    retarget = RobotRetarget(
        model_path=str(XML_PATH),
        keypoint_path=str(keypoints_path),
        ik_match_table={"hips_mean": ["hips_sphere", 1.0, 0.0]},
        contact_map=contact_map,
        contact_task_mode="legacy_hold",
        heel_toe_height_difference_cost=5.0,
        heel_toe_height_difference_ramp_seconds=0.04,
    )
    height_targets = {item["side"]: item for item in retarget.heel_toe_height_targets}
    np.testing.assert_allclose(
        height_targets["left"]["weights"], [0.0, 0.5, 1.0, 0.5, 0.0]
    )
    np.testing.assert_allclose(height_targets["right"]["weights"], 0.0)

    retarget.update_targets(1)
    left_task = height_targets["left"]["task"]
    assert left_task.cost[0] == pytest.approx(5.0 * np.sqrt(0.5))
    ankle_id = mujoco.mj_name2id(
        retarget.model, mujoco.mjtObj.mjOBJ_JOINT, "l_ankle_pitch_joint"
    )
    qpos = retarget.configuration.data.qpos.copy()
    qpos[int(retarget.model.jnt_qposadr[ankle_id])] += 0.15
    retarget.configuration.update(qpos)
    error = left_task.compute_error(retarget.configuration)
    jacobian = left_task.compute_jacobian(retarget.configuration)
    assert abs(error[0]) > 1.0e-5
    assert jacobian.shape == (1, retarget.model.nv)
    assert jacobian[0, 2] == pytest.approx(0.0, abs=1.0e-12)

    retarget.update_targets(0)
    assert left_task.cost[0] == 0.0
    np.testing.assert_array_equal(
        left_task.compute_error(retarget.configuration), np.zeros(1)
    )


def test_paired_support_flat_to_toe_off_and_weight_ramp(tmp_path: Path) -> None:
    """平足同时锁 heel/toe；toe-off 只保留 toe，并平滑改变二次代价。"""
    keypoints_path = tmp_path / "keypoints.pkl"
    contacts = ["left_foot_end", "left_toe", "right_foot_end", "right_toe"]
    states = np.asarray(
        [
            [False, False, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [False, True, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    write_keypoints(keypoints_path, contacts, states)
    contact_map = {
        name: {
            "frame_name": (
                f"{name}_link" if name.endswith("toe") else f"{name}_link"
            ),
            "source_keypoint_name": (
                f"{name}_link" if name.endswith("toe") else f"{name}_link"
            ),
            "position_cost": 120.0,
        }
        for name in contacts
    }
    retarget = RobotRetarget(
        model_path=str(XML_PATH),
        keypoint_path=str(keypoints_path),
        ik_match_table={"hips_mean": ["hips_sphere", 1.0, 0.0]},
        contact_map=contact_map,
        contact_task_mode="paired_support",
        contact_weight_ramp_seconds=0.04,
        post_ik_ground_bodies=[value["frame_name"] for value in contact_map.values()],
        post_ik_ground_clearance=0.002,
        post_ik_ground_mode="support_project",
    )
    targets = {item["contact_name"]: item for item in retarget.contact_targets}
    np.testing.assert_array_equal(
        targets["left_toe"]["paired_modes"], [0, 2, 2, 1, 0]
    )
    np.testing.assert_allclose(
        targets["left_foot_end"]["paired_weights"], [0.0, 0.5, 1.0, 0.5, 0.0]
    )
    np.testing.assert_allclose(
        targets["left_toe"]["paired_weights"], [0.0, 0.5, 1.0, 1.0, 0.5]
    )
    heel_target = targets["left_foot_end"]["paired_lock_positions"][1]
    toe_target = targets["left_toe"]["paired_lock_positions"][1]
    assert heel_target[2] == pytest.approx(0.002)
    assert toe_target[2] == pytest.approx(0.002)
    assert np.linalg.norm(toe_target[:2] - heel_target[:2]) == pytest.approx(
        retarget.paired_support_summary["feet"]["left"]["heel_toe_span_m"]
    )
    # 第 4 帧已经离开支撑，但权重仍在渐出，目标必须继续持有上一锁点。
    assert targets["left_foot_end"]["paired_lock_positions"][4, 2] == pytest.approx(
        0.002
    )
    assert targets["left_toe"]["paired_lock_positions"][4, 2] == pytest.approx(0.002)
    precontact = {
        item["body_name"]: item for item in retarget.paired_precontact_targets
    }
    np.testing.assert_allclose(
        precontact["left_foot_end_link"]["weights"], [1.0, 0.5, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        precontact["left_toe_link"]["weights"], [1.0, 0.5, 0.0, 0.0, 0.0]
    )
    assert np.allclose(
        precontact["left_foot_end_link"]["positions"][:, 2], 0.002
    )
    # 4 个常规足点任务和 4 个仅 Z 方向的接触前任务常驻 QP，权重连续变化，
    # 因此支撑状态切换时求解矩阵维数也不会变化。
    assert len(retarget.update_targets(2)) == len(retarget.base_tasks) + 8
    assert set(retarget.current_ground_body_ids) == {
        mujoco.mj_name2id(
            retarget.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_end_link"
        ),
        mujoco.mj_name2id(
            retarget.model, mujoco.mjtObj.mjOBJ_BODY, "left_toe_link"
        ),
    }
    retarget.update_targets(3)
    assert retarget.current_ground_body_ids == [
        mujoco.mj_name2id(
            retarget.model, mujoco.mjtObj.mjOBJ_BODY, "left_toe_link"
        )
    ]


def test_missing_explicit_contact_is_error(tmp_path: Path) -> None:
    keypoints_path = tmp_path / "keypoints.pkl"
    write_keypoints(keypoints_path, ["left_foot_end"], np.zeros((1, 1), dtype=bool))
    with pytest.raises(ValueError, match="source contact not found"):
        RobotRetarget(
            model_path=str(XML_PATH),
            keypoint_path=str(keypoints_path),
            ik_match_table={"hips_mean": ["hips_sphere", 1.0, 0.0]},
            contact_map={
                "missing": {
                    "frame_name": "left_foot_end_link",
                    "source_keypoint_name": "left_foot_end_link",
                }
            },
        )


def test_legacy_contact_links_still_work(tmp_path: Path) -> None:
    keypoints_path = tmp_path / "keypoints.pkl"
    write_keypoints(
        keypoints_path,
        ["left_foot_end", "right_foot_end"],
        np.zeros((2, 2), dtype=bool),
    )
    retarget = RobotRetarget(
        model_path=str(XML_PATH),
        keypoint_path=str(keypoints_path),
        ik_match_table={"hips_mean": ["hips_sphere", 1.0, 0.0]},
        contact_body_names=["left_foot_end_link", "right_foot_end_link"],
    )
    assert len(retarget.contact_targets) == 2
    assert len(retarget.update_targets(0)) == len(retarget.base_tasks) + 2
