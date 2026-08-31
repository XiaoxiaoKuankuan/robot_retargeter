"""BUMI3 四库正式批处理的坐标合同与发布门禁测试。

测试在 pytest 临时目录构造最小标准化 SMPL-X NPZ，验证批处理只接受明确声明为
右手 Z-up 米制、且记录了 Y-up 到 Z-up 历史转换的 30 Hz 数据；同时覆盖根倾角
计算与整库分布门禁，确保少量真实倒立动作不会拒绝发布，而整库约 90 度躺倒会
稳定失败。这里不启动 SMPL-X、Mink 或 MuJoCo 全流水线，集成 smoke 由正式脚本
在服务器的独立测试输出目录完成。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retarget_bumi3_full_dataset import (
    EXPECTED_COORDINATE_SYSTEM,
    audit_motion,
    distribution_passes,
    distribution_summary,
    sampled_root_tilt_degrees,
)


def write_motion(path: Path, coordinate_system: str = EXPECTED_COORDINATE_SYSTEM) -> None:
    """写入三帧直立、字段完整的标准化 SMPL-X 测试动作。"""
    np.savez(
        path,
        root_orient=np.tile(
            np.asarray([[np.pi / 2.0, 0.0, 0.0]], dtype=np.float32), (3, 1)
        ),
        pose_body=np.zeros((3, 63), dtype=np.float32),
        trans=np.zeros((3, 3), dtype=np.float32),
        betas=np.zeros(16, dtype=np.float32),
        mocap_frame_rate=np.asarray(30.0, dtype=np.float32),
        coordinate_system=np.asarray(coordinate_system),
        source_coordinate_system=np.asarray("right_handed_y_up_metric"),
        coordinate_transform=np.asarray(
            "rotate_global_root_and_translation_plus_90deg_about_x"
        ),
    )


def test_audit_motion_accepts_explicit_z_up_contract(tmp_path: Path) -> None:
    motion_path = tmp_path / "sample.npz"
    write_motion(motion_path)
    task = audit_motion(motion_path, "aistpp", expected_fps=30.0)
    assert task.frames == 3
    assert task.coordinate_system == EXPECTED_COORDINATE_SYSTEM
    assert task.source_root_tilt_median_deg == pytest.approx(0.0, abs=1.0e-5)
    assert len(task.source_sha256) == 64


def test_audit_motion_rejects_ambiguous_or_y_up_input(tmp_path: Path) -> None:
    motion_path = tmp_path / "wrong_axis.npz"
    write_motion(motion_path, coordinate_system="right_handed_y_up_metric")
    with pytest.raises(ValueError, match="正式输入必须明确"):
        audit_motion(motion_path, "aistpp", expected_fps=30.0)


def test_root_tilt_and_distribution_gate_detect_systematic_lie_down() -> None:
    upright = sampled_root_tilt_degrees(
        np.asarray([[np.pi / 2.0, 0.0, 0.0]], dtype=np.float64)
    )
    lying = sampled_root_tilt_degrees(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    )
    assert upright[0] == pytest.approx(0.0, abs=1.0e-6)
    assert lying[0] == pytest.approx(90.0, abs=1.0e-6)
    assert distribution_passes(distribution_summary([8.0] * 999 + [90.0]))
    assert not distribution_passes(distribution_summary([90.0] * 1000))
