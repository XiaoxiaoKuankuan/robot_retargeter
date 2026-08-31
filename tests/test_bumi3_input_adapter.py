"""BUMI3 流水线 SMPL/SMPL-X 输入适配器的纯单元测试。

本文件构造最小 NPZ，不加载人体模型，专门验证字段优先级、默认帧率、姿态维度、
平移尺度、Y-up 到 Z-up 的显式旋转以及真实时间重采样。测试覆盖仓库旧标准字段、
AMASS 风格 ``poses`` 与四集合实际使用的 ``pose + transl``，确保格式兼容改动
不会依赖某个真实数据文件偶然存在的额外元数据。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smpl_replay import (
    compute_contact_height_offsets,
    fit_stable_support_floor_height,
    load_motion_arrays,
    offset_keypoints_by_contact_height,
    stabilize_contact_states,
)


def base_arrays(frame_count: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回零根旋转、零 body 姿态和可辨识平移。"""
    root = np.zeros((frame_count, 3), dtype=np.float32)
    body = np.zeros((frame_count, 63), dtype=np.float32)
    trans = np.zeros((frame_count, 3), dtype=np.float32)
    trans[:, 0] = np.arange(frame_count)
    return root, body, trans


def test_standard_fields_and_default_fps(tmp_path: Path) -> None:
    root, body, trans = base_arrays()
    path = tmp_path / "standard.npz"
    np.savez(path, trans=trans, root_orient=root, pose_body=body)
    motion, gender, fps = load_motion_arrays(path, up_axis="z")
    assert fps == 30.0
    assert gender == "neutral"
    assert motion["trans"].shape == (3, 3)
    assert motion["pose_body"].shape == (3, 63)
    assert len(motion["source_motion_sha256"]) == 64


@pytest.mark.parametrize("translation_field", ["trans", "transl"])
def test_poses_72_translation_aliases(tmp_path: Path, translation_field: str) -> None:
    root, body, trans = base_arrays()
    path = tmp_path / f"poses_{translation_field}.npz"
    payload = {"poses": np.concatenate([root, body, np.zeros((3, 6))], axis=1)}
    payload[translation_field] = trans
    np.savez(path, **payload)
    motion, _gender, _fps = load_motion_arrays(path, up_axis="z")
    np.testing.assert_allclose(motion["root_orient"], root)
    np.testing.assert_allclose(motion["pose_body"], body)


def test_music_four_set_pose_and_transl(tmp_path: Path) -> None:
    root, body, trans = base_arrays()
    path = tmp_path / "music.npz"
    np.savez(path, pose=np.concatenate([root, body], axis=1), transl=trans, fps=30.0)
    motion, _gender, fps = load_motion_arrays(
        path, model_type_override="smplx", up_axis="z", target_fps=50.0
    )
    assert fps == 50.0
    assert motion["surface_model_type"] == "smplx"
    assert motion["trans"].shape[0] == 4


def test_invalid_pose_dimension(tmp_path: Path) -> None:
    path = tmp_path / "short_pose.npz"
    np.savez(path, poses=np.zeros((2, 65)), trans=np.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"D>=66"):
        load_motion_arrays(path, up_axis="z")


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan])
def test_invalid_translation_scale(tmp_path: Path, scale: float) -> None:
    root, body, trans = base_arrays()
    path = tmp_path / "scale.npz"
    np.savez(path, trans=trans, root_orient=root, pose_body=body)
    with pytest.raises(ValueError, match="translation_scale"):
        load_motion_arrays(path, translation_scale=scale, up_axis="z")


def test_explicit_y_up_to_z_up(tmp_path: Path) -> None:
    root, body, trans = base_arrays(frame_count=1)
    trans[0] = [0.0, 1.0, 0.0]
    path = tmp_path / "y_up.npz"
    np.savez(path, trans=trans, root_orient=root, pose_body=body)
    motion, _gender, _fps = load_motion_arrays(path, up_axis="y")
    np.testing.assert_allclose(motion["trans"][0], [0.0, 0.0, 1.0], atol=1.0e-6)
    np.testing.assert_allclose(motion["root_orient"][0], [np.pi / 2, 0.0, 0.0], atol=1.0e-6)


def test_sequence_floor_initializes_height_before_first_contact() -> None:
    """动作开头无接触时，应沿用全序列地板而不是错误地使用世界零高度。"""
    keypoints = np.zeros((4, 1, 3), dtype=np.float32)
    keypoints[:, 0, 2] = -1.25
    contact_positions = np.zeros((4, 1, 3), dtype=np.float32)
    contact_positions[:, 0, 2] = -1.25
    contact_states = np.asarray([[False], [False], [True], [True]], dtype=bool)
    offsets = compute_contact_height_offsets(
        keypoints=keypoints,
        keypoint_names=["left_foot_end_link"],
        contact_names=("left_foot_end",),
        contact_positions=contact_positions,
        contact_states=contact_states,
        initial_height=-1.25,
    )
    np.testing.assert_allclose(offsets, -1.25)


def test_disabled_dynamic_height_offset_uses_one_sequence_constant() -> None:
    """关闭动态偏移后，只能施加与接触开关无关的全序列恒定 Z 标定。"""
    keypoints = np.zeros((4, 2, 3), dtype=np.float32)
    keypoints[:, :, 2] = np.asarray([[0.4], [0.3], [0.2], [0.1]])
    contact_positions = np.zeros((4, 2, 3), dtype=np.float32)
    contact_positions[:, 0, 2] = -0.5
    contact_positions[:, 1, 2] = 1.0
    contact_states = np.asarray(
        [[True, False], [False, True], [True, True], [False, False]], dtype=bool
    )
    adjusted, offsets = offset_keypoints_by_contact_height(
        keypoints=keypoints,
        keypoint_names=["left_foot_end_link", "left_hand"],
        contact_names=("left_foot_end", "left_hand"),
        contact_positions=contact_positions,
        contact_states=contact_states,
        initial_height=-0.75,
        dynamic_offset_enabled=False,
    )
    expected = keypoints.copy()
    expected[:, :, 2] += 0.75
    np.testing.assert_array_equal(adjusted, expected)
    np.testing.assert_array_equal(offsets, np.full(4, -0.75, dtype=np.float32))
    np.testing.assert_allclose(
        np.diff(adjusted, axis=0), np.diff(keypoints, axis=0), atol=1.0e-7
    )


def test_dynamic_height_offset_cannot_read_unlisted_hand_keypoint() -> None:
    """动态高度只读取显式传入的足点，极低手点不能改变整机偏移。"""
    keypoints = np.zeros((3, 2, 3), dtype=np.float32)
    keypoints[:, 0, 2] = [0.50, 0.48, 0.46]
    keypoints[:, 1, 2] = [-10.0, -20.0, -30.0]
    foot_positions = keypoints[:, :1, :].copy()
    foot_states = np.ones((3, 1), dtype=bool)
    _adjusted, offsets = offset_keypoints_by_contact_height(
        keypoints=keypoints,
        keypoint_names=["left_foot_end_link", "left_hand"],
        contact_names=("left_foot_end",),
        contact_positions=foot_positions,
        contact_states=foot_states,
        dynamic_offset_enabled=True,
    )
    np.testing.assert_allclose(offsets, [0.50, 0.48, 0.46])


def test_stable_support_floor_ignores_fast_points_and_sparse_low_outliers() -> None:
    """地板应来自低速密集支撑簇，而不是快速摆动点或少量极低异常值。"""
    positions = np.zeros((12, 4, 3), dtype=np.float64)
    positions[:, :, 2] = np.asarray(
        [
            [0.401, 0.398, 0.75, 0.80],
            [0.399, 0.402, 0.76, 0.82],
            [0.400, 0.401, 0.74, 0.81],
            [0.403, 0.397, 0.73, 0.83],
            [0.398, 0.400, 0.72, 0.84],
            [0.401, 0.399, 0.71, 0.85],
            [0.400, 0.402, 0.70, 0.86],
            [0.399, 0.401, 0.69, 0.87],
            [0.402, 0.398, 0.68, 0.88],
            [0.400, 0.399, 0.67, 0.89],
            [-2.0, 0.401, 0.66, 0.90],
            [-1.8, 0.400, 0.65, 0.91],
        ]
    )
    speeds = np.full((12, 4), 0.05, dtype=np.float64)
    speeds[:, 2:] = 0.8
    floor, report = fit_stable_support_floor_height(
        contact_positions=positions,
        contact_speeds=speeds,
        reference_indices=np.arange(4, dtype=np.int32),
        speed_threshold_mps=0.2,
        inlier_tolerance_m=0.04,
        minimum_samples=8,
    )
    assert floor == pytest.approx(0.400, abs=0.002)
    assert report["method"] == "stable_support_dense_median"
    assert report["stable_sample_count"] == 24
    assert report["inlier_sample_count"] == 22


def test_stable_support_floor_rejects_too_few_slow_samples() -> None:
    positions = np.zeros((3, 4, 3), dtype=np.float64)
    speeds = np.ones((3, 4), dtype=np.float64)
    speeds[0, :2] = 0.1
    with pytest.raises(ValueError, match="低速足点不足"):
        fit_stable_support_floor_height(
            contact_positions=positions,
            contact_speeds=speeds,
            reference_indices=np.arange(4, dtype=np.int32),
            speed_threshold_mps=0.2,
            inlier_tolerance_m=0.04,
            minimum_samples=8,
        )


def test_contact_hysteresis_rejects_single_frame_chatter() -> None:
    """单帧超退出阈值不能断开接触，进入接触也必须连续确认。"""
    speeds = np.asarray(
        [[0.8], [0.2], [0.2], [0.2], [0.8], [0.2], [0.8], [0.8]],
        dtype=np.float64,
    )
    heights = np.zeros_like(speeds)
    states = stabilize_contact_states(
        contact_speeds=speeds,
        relative_heights=heights,
        enter_velocity=0.35,
        exit_velocity=0.60,
        enter_height=0.025,
        exit_height=0.05,
        enter_confirmation_frames=2,
        exit_confirmation_frames=2,
        minimum_contact_frames=2,
        minimum_swing_frames=2,
    )
    np.testing.assert_array_equal(
        states[:, 0], [False, True, True, True, True, True, True, False]
    )
