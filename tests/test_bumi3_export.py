"""BUMI3 MuJoCo qpos 到 IsaacLab Mimic 数据契约的导出测试。

测试使用真实 BUMI3 MJCF 的三帧合法 qpos，但在临时目录生成 CSV、JSON 与
NPZ。它检查 CSV ``xyzw`` 根四元数能还原为 MuJoCo ``wxyz``，Isaac 的 21
关节顺序通过名称而不是 MuJoCo 原生 qpos 顺序提取，22 个 body shape 完整，
并专门构造 ``q/-q`` 序列验证四元数符号翻转不会制造虚假 body 角速度尖峰。
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from bumi3_common import (
    BUMI3_JOINT_NAMES,
    joint_qpos_addresses,
    load_yaml,
    quaternion_angular_velocity_wxyz,
    sha256_file,
)
from export_bumi3_mimic_npz import export_mimic, load_csv_qpos


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config/robot/bumi3.yaml"
XML_PATH = REPOSITORY_ROOT / "asset/robot/bumi3/mjcf/bumi3_retarget.xml"


def test_quaternion_sign_flip_has_no_angular_spike() -> None:
    quaternions = np.asarray(
        [[[1.0, 0.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]
    )
    angular_velocity = quaternion_angular_velocity_wxyz(quaternions, fps=50.0)
    np.testing.assert_allclose(angular_velocity, 0.0, atol=1.0e-10)


def test_export_shapes_orders_and_xyzw_conversion(tmp_path: Path) -> None:
    target_fps = float(load_yaml(CONFIG_PATH)["output"]["target_fps"])
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    qpos = np.broadcast_to(model.qpos0, (3, model.nq)).copy()
    qpos[:, 2] = 0.65
    qpos[:, 3:7] = [1.0, 0.0, 0.0, 0.0]
    addresses = joint_qpos_addresses(model, BUMI3_JOINT_NAMES)
    for joint_index, name in enumerate(BUMI3_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower, upper = model.jnt_range[joint_id]
        qpos[:, addresses[name]] = lower + (0.35 + 0.01 * joint_index) * (upper - lower)
    csv_values = qpos.copy()
    csv_values[:, 3:7] = qpos[:, [4, 5, 6, 3]]
    csv_path = tmp_path / "motion_bumi3.csv"
    np.savetxt(csv_path, csv_values, delimiter=",")
    metadata_path = tmp_path / "motion_bumi3.meta.json"
    metadata_path.write_text(
        json.dumps(
            {
                "robot": "bumi3",
                "source_motion": "unit-test",
                "fps": target_fps,
                "num_frames": 3,
                "qpos_size": model.nq,
                "robot_xml_sha256": sha256_file(XML_PATH),
                "config_sha256": sha256_file(CONFIG_PATH),
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "motion.npz"
    export_mimic(csv_path, metadata_path, CONFIG_PATH, output_path)

    reconstructed = load_csv_qpos(csv_path, model)
    np.testing.assert_allclose(reconstructed[:, 3:7], qpos[:, 3:7])
    with np.load(output_path) as payload:
        assert payload["joint_pos"].shape == (3, 21)
        assert payload["joint_vel"].shape == (3, 21)
        assert payload["body_pos_w"].shape == (3, 22, 3)
        assert payload["body_quat_w"].shape == (3, 22, 4)
        assert payload["body_lin_vel_w"].shape == (3, 22, 3)
        assert payload["body_ang_vel_w"].shape == (3, 22, 3)
        np.testing.assert_allclose(
            payload["joint_pos"][0],
            [qpos[0, addresses[name]] for name in BUMI3_JOINT_NAMES],
        )
        assert payload["quaternion_order"].item() == "wxyz"
        assert payload["anchor_body_name"].item() == "waist_yaw_link"
