#!/usr/bin/env python3
"""整理同一批 BUMI3 重定向结果，生成可直接对照播放的两套 CSV。

本脚本不执行 IK，也不对轨迹做平滑、插值、限位裁剪或高度修正。它只完成三件事：

1. 以数据集目录中的 NPZ 为唯一清单，要求当前仓库 CSV 与原始 GMR PKL 一一对应；
2. 原样复制当前仓库 CSV，并把 GMR 最新流水线 PKL 的 ``root_pos/root_rot/dof_pos`` 按 GMR
   XML 中的关节名称和 ``jnt_qposadr`` 封装成相同的 CSV 播放格式；
3. 核对两种方法的帧数、帧率、有限值、四元数、模型和配置哈希，写出可追溯的
   ``comparison_manifest.json``。

GMR PKL 的根四元数与本仓库 CSV 均使用 ``xyzw``；MuJoCo 播放器加载时再转成
``wxyz``。两套 BUMI3 XML 的 21 个关节名称相同但 qpos 排列不同，因此本脚本
禁止直接复制关节列，必须逐关节按名称落到 GMR XML 的 qpos 地址。生成的目录仅
用于离线 MuJoCo 运动学效果比较，不代表动力学可执行性或实机安全性。
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bumi3_common import sha256_file
from export_bumi3_mimic_npz import load_csv_qpos


def parse_args() -> argparse.Namespace:
    """解析对比整理所需的输入、资产和输出路径。"""
    parser = argparse.ArgumentParser(description="整理 BUMI3 当前方法与原始 GMR 对比轨迹")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/music_smpl_4set"),
        help="包含本次固定选择 NPZ 的数据集根目录",
    )
    parser.add_argument(
        "--current-motion-dir",
        type=Path,
        default=Path("output_data/robot_motion"),
        help="当前仓库方法的 CSV/metadata 目录",
    )
    parser.add_argument(
        "--current-robot-xml",
        type=Path,
        default=Path("asset/robot/bumi3/mjcf/bumi3_retarget.xml"),
        help="当前仓库方法对应的 BUMI3 XML",
    )
    parser.add_argument(
        "--gmr-pkl-dir",
        type=Path,
        default=Path("output_data/comparison/gmr_latest_pkl"),
        help="GMR 最新 temporal/trajectory/root-height 流水线输出 PKL 根目录",
    )
    parser.add_argument("--gmr-robot-xml", type=Path, required=True)
    parser.add_argument("--gmr-ik-config", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output_data/comparison"),
        help="整理后的两套 CSV 与清单输出根目录",
    )
    parser.add_argument("--expected-count", type=int, default=10)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    """把 numpy/Path 标量递归转换为标准 JSON 类型。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """以临时文件替换方式写 JSON，避免中断时留下半份清单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _single_free_joint_contract(model: mujoco.MjModel) -> tuple[int, str]:
    """返回唯一 free joint 的 qpos 地址和名称，并要求 CSV 根块从第 0 列开始。"""
    joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(joint_ids) != 1:
        raise ValueError(f"模型必须恰有一个 free joint: actual={len(joint_ids)}")
    joint_id = joint_ids[0]
    address = int(model.jnt_qposadr[joint_id])
    if address != 0:
        raise ValueError(f"播放器 CSV 要求 free joint qpos 地址为 0: actual={address}")
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    return address, str(name)


def _hinge_qpos_contract(model: mujoco.MjModel) -> dict[str, int]:
    """提取所有单自由度 hinge 的名称到 qpos 地址映射。"""
    contract: dict[str, int] = {}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None or name in contract:
            raise ValueError(f"hinge 关节名称缺失或重复: joint_id={joint_id}, name={name}")
        contract[name] = int(model.jnt_qposadr[joint_id])
    return contract


def _load_gmr_pickle(path: Path) -> dict[str, Any]:
    """读取并检查 GMR PKL 的核心轨迹合同。"""
    with path.open("rb") as file_handle:
        motion = pickle.load(file_handle)
    required = ("fps", "root_pos", "root_rot", "dof_pos", "dof_names", "quality")
    missing = [field for field in required if field not in motion]
    if missing:
        raise ValueError(f"GMR PKL 缺少字段: path={path}, missing={missing}")
    return motion


def _gmr_motion_to_csv_values(
    motion: dict[str, Any], model: mujoco.MjModel
) -> tuple[np.ndarray, dict[str, Any]]:
    """按名称把 GMR 轨迹封装为对应 XML 的 CSV qpos 列顺序。"""
    free_address, free_name = _single_free_joint_contract(model)
    hinge_addresses = _hinge_qpos_contract(model)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    dof_names = [str(name) for name in motion["dof_names"]]
    frame_count = root_pos.shape[0]
    expected_shapes = {
        "root_pos": (frame_count, 3),
        "root_rot": (frame_count, 4),
        "dof_pos": (frame_count, len(dof_names)),
    }
    actual_shapes = {
        "root_pos": root_pos.shape,
        "root_rot": root_xyzw.shape,
        "dof_pos": dof_pos.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"GMR 轨迹 shape 不一致: expected={expected_shapes}, actual={actual_shapes}"
        )
    if len(dof_names) != len(set(dof_names)):
        raise ValueError("GMR dof_names 包含重复名称")
    if set(dof_names) != set(hinge_addresses):
        raise ValueError(
            "GMR PKL 与 XML 的 hinge 名称集合不同: "
            f"missing={sorted(set(hinge_addresses) - set(dof_names))}, "
            f"extra={sorted(set(dof_names) - set(hinge_addresses))}"
        )
    if not all(np.all(np.isfinite(values)) for values in (root_pos, root_xyzw, dof_pos)):
        raise ValueError("GMR 轨迹包含 NaN 或 Inf")
    quaternion_norms = np.linalg.norm(root_xyzw, axis=1)
    quaternion_norm_error = float(np.max(np.abs(quaternion_norms - 1.0)))
    if quaternion_norm_error > 1.0e-5:
        raise ValueError(f"GMR 根四元数未归一化: max_error={quaternion_norm_error}")

    csv_values = np.zeros((frame_count, model.nq), dtype=np.float64)
    csv_values[:, free_address : free_address + 3] = root_pos
    csv_values[:, free_address + 3 : free_address + 7] = root_xyzw
    dof_index = {name: index for index, name in enumerate(dof_names)}
    for name, address in hinge_addresses.items():
        csv_values[:, address] = dof_pos[:, dof_index[name]]

    max_joint_limit_excess = 0.0
    for name, address in hinge_addresses.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if not bool(model.jnt_limited[joint_id]):
            continue
        lower, upper = model.jnt_range[joint_id]
        values = csv_values[:, address]
        excess = np.maximum(lower - values, values - upper)
        max_joint_limit_excess = max(max_joint_limit_excess, float(np.max(excess)))
    return csv_values, {
        "free_joint_name": free_name,
        "free_joint_qpos_address": free_address,
        "hinge_joint_qpos_addresses": hinge_addresses,
        "gmr_dof_names": dof_names,
        "root_quaternion_norm_error_max": quaternion_norm_error,
        "max_joint_limit_excess_rad": max(0.0, max_joint_limit_excess),
    }


def _copy_current_pair(csv_path: Path, metadata_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """复制当前方法的 CSV/metadata，不重写其中任何轨迹数值。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / csv_path.name
    output_metadata = output_dir / metadata_path.name
    shutil.copy2(csv_path, output_csv)
    shutil.copy2(metadata_path, output_metadata)
    return output_csv, output_metadata


def main() -> None:
    """建立 10 条同输入、同帧率、方法和资产身份均可追溯的比较产物。"""
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    current_motion_dir = args.current_motion_dir.expanduser().resolve()
    current_robot_xml = args.current_robot_xml.expanduser().resolve()
    gmr_pkl_dir = args.gmr_pkl_dir.expanduser().resolve()
    gmr_robot_xml = args.gmr_robot_xml.expanduser().resolve()
    gmr_ik_config = args.gmr_ik_config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    required_paths = (
        dataset_root,
        current_motion_dir,
        current_robot_xml,
        gmr_pkl_dir,
        gmr_robot_xml,
        gmr_ik_config,
    )
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"比较输入不存在: {missing_paths}")

    # 播放器按扁平 CSV 文件名排序；清单必须使用同一规则，确保两个窗口的数字键
    # 以及 manifest 中的 index 永远指向同一条动作。
    source_paths = sorted(dataset_root.rglob("*.npz"), key=lambda path: path.stem)
    if len(source_paths) != args.expected_count:
        raise ValueError(
            f"数据集清单数量不符: expected={args.expected_count}, actual={len(source_paths)}"
        )
    duplicate_stems = sorted(
        {path.stem for path in source_paths if sum(other.stem == path.stem for other in source_paths) > 1}
    )
    if duplicate_stems:
        raise ValueError(f"扁平播放器输出存在重名 stem: {duplicate_stems}")
    gmr_summary_path = gmr_pkl_dir / "_conversion_summary.json"
    if not gmr_summary_path.is_file():
        raise FileNotFoundError(f"缺少 GMR 批次摘要: {gmr_summary_path}")
    gmr_summary = json.loads(gmr_summary_path.read_text(encoding="utf-8"))
    pipeline_config = dict(gmr_summary.get("pipeline_config", {}))
    latest_contract = {
        "temporal_ik": True,
        "project_wrist_targets": True,
        "trajectory_optimization": True,
        "root_height_optimization": True,
        "legacy_root_alignment": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": pipeline_config.get(key)}
        for key, expected in latest_contract.items()
        if pipeline_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"GMR 输出不是最新 temporal/trajectory/foot-root 流程: {mismatches}")

    current_model = mujoco.MjModel.from_xml_path(str(current_robot_xml))
    gmr_model = mujoco.MjModel.from_xml_path(str(gmr_robot_xml))
    _single_free_joint_contract(current_model)
    _single_free_joint_contract(gmr_model)
    current_output_dir = output_root / "robot_retargeter"
    gmr_output_dir = output_root / "gmr_latest"
    motions: list[dict[str, Any]] = []
    current_config_hashes: set[str] = set()
    current_xml_hashes: set[str] = set()
    fps_values: set[float] = set()

    for source_path in source_paths:
        relative_source = source_path.relative_to(dataset_root)
        stem = source_path.stem
        filename = f"{stem}_bumi3.csv"
        current_csv = current_motion_dir / filename
        current_metadata = current_csv.with_suffix(".meta.json")
        gmr_pkl = (gmr_pkl_dir / relative_source).with_suffix(".pkl")
        for required_path in (current_csv, current_metadata, gmr_pkl):
            if not required_path.is_file():
                raise FileNotFoundError(f"动作缺少对应产物: {required_path}")

        current_meta = json.loads(current_metadata.read_text(encoding="utf-8"))
        recorded_source = Path(str(current_meta.get("source_motion", ""))).resolve()
        if recorded_source != source_path.resolve():
            raise ValueError(
                f"当前方法 metadata 源动作不匹配: expected={source_path}, actual={recorded_source}"
            )
        current_qpos = load_csv_qpos(current_csv, current_model)
        if current_qpos.shape[0] != int(current_meta.get("num_frames", -1)):
            raise ValueError(f"当前方法 CSV 与 metadata 帧数不一致: {current_csv}")
        current_fps = float(current_meta.get("fps", 0.0))

        motion = _load_gmr_pickle(gmr_pkl)
        quality = dict(motion["quality"])
        motion_pipeline_config = dict(quality.get("pipeline_config", {}))
        motion_pipeline_mismatches = {
            key: {"expected": expected, "actual": motion_pipeline_config.get(key)}
            for key, expected in latest_contract.items()
            if motion_pipeline_config.get(key) != expected
        }
        if motion_pipeline_mismatches:
            raise ValueError(
                f"GMR PKL 不是最新流水线: path={gmr_pkl}, mismatches={motion_pipeline_mismatches}"
            )
        root_height_method = quality.get("root_height", {}).get("method")
        if root_height_method != "foot_contact_bounded_qp":
            raise ValueError(
                "GMR 最新结果必须只使用足底 mesh Root-Z QP: "
                f"path={gmr_pkl}, actual={root_height_method}"
            )
        gmr_csv_values, gmr_contract = _gmr_motion_to_csv_values(motion, gmr_model)
        gmr_fps = float(motion["fps"])
        if current_qpos.shape[0] != gmr_csv_values.shape[0]:
            raise ValueError(
                "两种方法帧数不一致: "
                f"motion={relative_source}, current={current_qpos.shape[0]}, "
                f"gmr={gmr_csv_values.shape[0]}"
            )
        if abs(current_fps - gmr_fps) > 1.0e-9:
            raise ValueError(
                f"两种方法 fps 不一致: motion={relative_source}, current={current_fps}, gmr={gmr_fps}"
            )

        copied_csv, copied_metadata = _copy_current_pair(
            current_csv, current_metadata, current_output_dir
        )
        gmr_output_dir.mkdir(parents=True, exist_ok=True)
        gmr_csv = gmr_output_dir / filename
        np.savetxt(gmr_csv, gmr_csv_values, delimiter=",", fmt="%.10f")
        gmr_meta = {
            "robot": "bumi3",
            "method": "gmr_latest_temporal_trajectory_foot_root_qp",
            "source_motion": str(source_path),
            "source_motion_sha256": sha256_file(source_path),
            "source_gmr_pkl": str(gmr_pkl),
            "source_gmr_pkl_sha256": sha256_file(gmr_pkl),
            "fps": gmr_fps,
            "num_frames": int(gmr_csv_values.shape[0]),
            "qpos_size": int(gmr_model.nq),
            "root_quaternion_order_in_csv": "xyzw",
            "robot_xml_path": str(gmr_robot_xml),
            "robot_xml_sha256": sha256_file(gmr_robot_xml),
            "ik_config_path": str(gmr_ik_config),
            "ik_config_sha256": sha256_file(gmr_ik_config),
            "pipeline_config": pipeline_config,
            "quality_acceptance": quality.get("acceptance", {}),
            "root_height_method": root_height_method,
            "root_height_diagnostics": quality.get("root_height", {}),
            "final_root_audit": quality.get("final_root_audit", {}),
            "serialization_only": True,
            "serialization_postprocessing": [],
            "qpos_contract": gmr_contract,
        }
        gmr_metadata = gmr_csv.with_suffix(".meta.json")
        _write_json(gmr_metadata, gmr_meta)

        current_config_hashes.add(str(current_meta.get("config_sha256", "")))
        current_xml_hashes.add(str(current_meta.get("robot_xml_sha256", "")))
        fps_values.add(current_fps)
        motions.append(
            {
                "index": len(motions) + 1,
                "dataset": relative_source.parts[0],
                "source": str(source_path),
                "source_relative": str(relative_source),
                "source_sha256": sha256_file(source_path),
                "frames": int(current_qpos.shape[0]),
                "fps": current_fps,
                "robot_retargeter_csv": str(copied_csv),
                "robot_retargeter_metadata": str(copied_metadata),
                "robot_retargeter_csv_sha256": sha256_file(copied_csv),
                "gmr_latest_csv": str(gmr_csv),
                "gmr_latest_metadata": str(gmr_metadata),
                "gmr_latest_csv_sha256": sha256_file(gmr_csv),
            }
        )

    if len(current_config_hashes) != 1 or "" in current_config_hashes:
        raise ValueError(f"当前方法配置 SHA 不唯一或缺失: {current_config_hashes}")
    if current_xml_hashes != {sha256_file(current_robot_xml)}:
        raise ValueError(
            "当前方法 metadata 的 XML SHA 与播放器 XML 不一致: "
            f"metadata={current_xml_hashes}, player={sha256_file(current_robot_xml)}"
        )
    if len(fps_values) != 1:
        raise ValueError(f"比较动作 fps 不唯一: {fps_values}")

    manifest = {
        "comparison_version": "bumi3_robot_retargeter_vs_gmr_latest_v2",
        "selection": {
            "rule": "沿用本仓库已下载并人工筛选的四个音乐 SMPL-X 集合全部 10 条",
            "count": len(motions),
            "dataset_counts": {
                dataset: sum(motion["dataset"] == dataset for motion in motions)
                for dataset in sorted({motion["dataset"] for motion in motions})
            },
        },
        "fairness_contract": {
            "same_source_npz": True,
            "same_frame_count_per_motion": True,
            "same_fps": sorted(fps_values),
            "extra_smoothing_during_packaging": False,
            "extra_height_correction_during_packaging": False,
            "important_boundary": (
                "两种方法使用各自匹配的 BUMI3 XML；资产 SHA 不同，因此结果比较同时包含"
                "算法配置与模型资产差异，不能解释为纯算法消融。"
            ),
        },
        "methods": {
            "robot_retargeter": {
                "motion_dir": str(current_output_dir),
                "config_sha256": next(iter(current_config_hashes)),
                "robot_xml_path": str(current_robot_xml),
                "robot_xml_sha256": sha256_file(current_robot_xml),
            },
            "gmr_latest": {
                "motion_dir": str(gmr_output_dir),
                "source_pkl_dir": str(gmr_pkl_dir),
                "pipeline_config": pipeline_config,
                "quality_write_policy": (
                    "strict_reject" if pipeline_config.get("strict_quality") else "allow_quality_failure"
                ),
                "ik_config_path": str(gmr_ik_config),
                "ik_config_sha256": sha256_file(gmr_ik_config),
                "robot_xml_path": str(gmr_robot_xml),
                "robot_xml_sha256": sha256_file(gmr_robot_xml),
            },
        },
        "total_frames_per_method": sum(motion["frames"] for motion in motions),
        "motions": motions,
    }
    manifest_path = output_root / "comparison_manifest.json"
    _write_json(manifest_path, manifest)
    print(
        f"BUMI3 对比轨迹已整理：motions={len(motions)}, "
        f"frames_per_method={manifest['total_frames_per_method']}, manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
