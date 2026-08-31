#!/usr/bin/env python3
"""从本地 BUMI3 权威资产生成本仓库专用的重定向模型。

脚本只读取用户指定的 BUMI3 源目录，复制 STL 网格，并从源 MJCF 派生
``bumi3_retarget.xml``。派生时保留质量、惯量、关节轴、执行器、碰撞、传感器
和 body 层级；允许的改动仅包括按 URDF 修正位置限位、修正相对 meshdir，以及
加入无质量、无碰撞的固定 marker body。髋/肩中心来自零位姿关节锚点，
头顶、脚跟/脚尖和手端来自实际 STL 几何估计，置信度不足时会报错或要求显式
override，绝不会写入经验猜测坐标。四个脚跟/脚尖 marker 使用与 G1 一致的不透明
红球，便于在普通轨迹播放器中直接判断平脚、悬空和穿地；其余 IK marker 保持透明。
脚本可重复执行并输出含来源 SHA 的 JSON 报告。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh
import yaml

from bumi3_common import BUMI3_JOINT_NAMES, BUMI3_MARKER_BODY_NAMES, sha256_file


TRANSPARENT_MARKER_GEOM_ATTRIBUTES = {
    "type": "sphere",
    "size": "0.005",
    "density": "0",
    "contype": "0",
    "conaffinity": "0",
    "group": "1",
    "rgba": "1 0 0 0",
}

VISIBLE_FOOT_MARKER_NAMES = {
    "left_foot_end_link",
    "left_toe_link",
    "right_foot_end_link",
    "right_toe_link",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 BUMI3 MuJoCo 重定向资产")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(os.environ["BUMI_SOURCE_DIR"]) if "BUMI_SOURCE_DIR" in os.environ else None,
        help="包含 mjcf/bumi3.xml、urdf/bumi.urdf、meshes/*.STL 的 BUMI3 目录",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("asset/robot/bumi3"))
    parser.add_argument(
        "--overrides",
        type=Path,
        default=Path("config/robot/bumi3_marker_overrides.yaml"),
        help="自动几何估计失败时的显式 marker 局部坐标覆盖",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有且内容不同的派生资产")
    return parser.parse_args()


def _load_overrides(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    markers = payload.get("markers", {})
    if not isinstance(markers, dict):
        raise ValueError(f"marker override 必须是映射: path={path}, field=markers")
    result: dict[str, list[float]] = {}
    for name, value in markers.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"marker override 必须是 3 个有限值: path={path}, field=markers.{name}, actual={value}"
            )
        result[str(name)] = array.tolist()
    return result


def _body_element(root: ET.Element, name: str) -> ET.Element:
    matches = [element for element in root.iter("body") if element.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"MJCF body 必须唯一: name={name}, actual_count={len(matches)}")
    return matches[0]


def _body_local_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    parent_name: str,
    world_position: np.ndarray,
) -> np.ndarray:
    parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parent_name)
    if parent_id < 0:
        raise ValueError(f"源 MJCF 缺少 parent body: {parent_name}")
    rotation = data.xmat[parent_id].reshape(3, 3)
    return rotation.T @ (world_position - data.xpos[parent_id])


def _mesh_vertices(path: Path) -> np.ndarray:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geometry for geometry in loaded.geometry.values()]
        if not geometries:
            raise ValueError(f"STL 没有几何体: {path}")
        loaded = trimesh.util.concatenate(geometries)
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 20:
        raise ValueError(f"STL 顶点无效: path={path}, actual_shape={vertices.shape}")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"STL 顶点包含 NaN/Inf: {path}")
    return vertices


def _estimate_foot_markers(vertices: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    spans = bounds_max - bounds_min
    z_min = float(np.quantile(vertices[:, 2], 0.01))
    sole_band_height = max(0.004, 0.05 * float(spans[2]))
    sole = vertices[vertices[:, 2] <= z_min + sole_band_height]
    if sole.shape[0] < 20:
        raise ValueError(f"{side} 脚底带顶点不足: actual={sole.shape[0]}, expected>=20")
    x_span = float(np.ptp(sole[:, 0]))
    y_span = float(np.ptp(sole[:, 1]))
    if x_span <= 1.2 * y_span:
        raise ValueError(
            f"{side} 脚部前后轴置信度不足: expected=x_span>1.2*y_span, actual={x_span:.6f}/{y_span:.6f}"
        )
    heel = np.array(
        [np.percentile(sole[:, 0], 2) + 0.003, np.median(sole[:, 1]), np.percentile(sole[:, 2], 5) + 0.001]
    )
    toe = np.array(
        [np.percentile(sole[:, 0], 98) - 0.003, np.median(sole[:, 1]), np.percentile(sole[:, 2], 5) + 0.001]
    )
    if toe[0] <= heel[0]:
        raise ValueError(f"{side} 脚尖不在脚跟前方: heel_x={heel[0]}, toe_x={toe[0]}")
    tolerance = 0.02
    for name, point in (("heel", heel), ("toe", toe)):
        if np.any(point < bounds_min - tolerance) or np.any(point > bounds_max + tolerance):
            raise ValueError(f"{side} {name} marker 超出脚 mesh 邻域: point={point.tolist()}")
    return heel, toe, {
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "sole_vertices": int(sole.shape[0]),
        "sole_band_height": sole_band_height,
        "x_span": x_span,
        "y_span": y_span,
    }


def _estimate_hand_marker(vertices: np.ndarray, side: str) -> tuple[np.ndarray, dict[str, Any]]:
    centered = vertices - np.median(vertices, axis=0)
    _u, _singular, vh = np.linalg.svd(centered, full_matrices=False)
    principal = vh[0]
    projection = centered @ principal
    low = vertices[projection <= np.percentile(projection, 2)]
    high = vertices[projection >= np.percentile(projection, 98)]
    low_center = np.median(low, axis=0)
    high_center = np.median(high, axis=0)
    marker = low_center if np.linalg.norm(low_center) >= np.linalg.norm(high_center) else high_center
    distance = float(np.linalg.norm(marker))
    if distance <= 0.05:
        raise ValueError(f"{side} 前臂远端估计过近: distance={distance:.6f}, expected>0.05")
    return marker, {
        "principal_axis": principal.tolist(),
        "low_center": low_center.tolist(),
        "high_center": high_center.tolist(),
        "distance_from_joint": distance,
    }


def _add_marker(parent: ET.Element, name: str, position: np.ndarray) -> None:
    body = ET.SubElement(
        parent,
        "body",
        {"name": name, "pos": " ".join(f"{float(value):.9g}" for value in position)},
    )
    attributes = dict(TRANSPARENT_MARKER_GEOM_ATTRIBUTES)
    if name in VISIBLE_FOOT_MARKER_NAMES:
        attributes["size"] = "0.01"
        attributes["rgba"] = "1 0 0 1"
    ET.SubElement(body, "geom", attributes)


def _urdf_joint_limits(urdf_path: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(urdf_path).getroot()
    result: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") not in {"revolute", "continuous"}:
            continue
        name = joint.get("name")
        limit = joint.find("limit")
        if not name or limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"URDF 关节缺少位置限位: path={urdf_path}, joint={name}")
        result[name] = (float(limit.get("lower")), float(limit.get("upper")))
    return result


def _write_if_allowed(path: Path, content: bytes, overwrite: bool) -> str:
    if path.exists():
        if path.read_bytes() == content:
            return "unchanged"
        if not overwrite:
            raise FileExistsError(f"目标已存在且内容不同；请使用 --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return "written"


def prepare_asset(source_dir: Path, output_dir: Path, overrides_path: Path, overwrite: bool) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_mjcf = source_dir / "mjcf" / "bumi3.xml"
    source_urdf = source_dir / "urdf" / "bumi.urdf"
    source_mesh_dir = source_dir / "meshes"
    missing = [path for path in (source_mjcf, source_urdf, source_mesh_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"BUMI3 源资产不完整: source={source_dir}, missing={missing}")
    mesh_paths = sorted(source_mesh_dir.glob("*.STL"), key=lambda path: path.name)
    if not mesh_paths:
        raise FileNotFoundError(f"BUMI3 源 mesh 为空: expected={source_mesh_dir}/*.STL")

    output_mesh_dir = output_dir / "meshes"
    output_mesh_dir.mkdir(parents=True, exist_ok=True)
    copied_meshes: list[dict[str, str]] = []
    for source in mesh_paths:
        destination = output_mesh_dir / source.name
        status = _write_if_allowed(destination, source.read_bytes(), overwrite)
        copied_meshes.append({"name": source.name, "status": status, "sha256": sha256_file(source)})

    tree = ET.parse(source_mjcf)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", "../meshes/")

    urdf_limits = _urdf_joint_limits(source_urdf)
    if set(urdf_limits) != set(BUMI3_JOINT_NAMES):
        raise ValueError(
            "URDF 驱动关节集合不符合 BUMI3 契约: "
            f"path={source_urdf}, missing={sorted(set(BUMI3_JOINT_NAMES)-set(urdf_limits))}, "
            f"extra={sorted(set(urdf_limits)-set(BUMI3_JOINT_NAMES))}"
        )
    mjcf_joints = {joint.get("name"): joint for joint in root.iter("joint") if joint.get("name")}
    limit_changes: list[dict[str, Any]] = []
    for name in BUMI3_JOINT_NAMES:
        joint = mjcf_joints.get(name)
        if joint is None:
            raise ValueError(f"MJCF 缺少 URDF 同名关节: path={source_mjcf}, joint={name}")
        raw_range = joint.get("range")
        if raw_range is None:
            raise ValueError(f"MJCF 关节缺少 range: path={source_mjcf}, joint={name}")
        actual = tuple(float(value) for value in raw_range.split())
        expected = urdf_limits[name]
        if len(actual) != 2:
            raise ValueError(f"MJCF range 必须含两个值: joint={name}, actual={raw_range}")
        if not np.allclose(actual, expected, atol=1.0e-9, rtol=0.0):
            joint.set("range", f"{expected[0]:.9g} {expected[1]:.9g}")
            limit_changes.append({"joint": name, "from": list(actual), "to": list(expected)})

    for marker_name in BUMI3_MARKER_BODY_NAMES:
        for existing in [element for element in root.iter("body") if element.get("name") == marker_name]:
            parent = next(parent for parent in root.iter() if existing in list(parent))
            parent.remove(existing)

    model = mujoco.MjModel.from_xml_path(str(source_mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_world = {
        name: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()
        for name in ("l_leg_pitch_link", "r_leg_pitch_link", "l_arm_pitch_link", "r_arm_pitch_link")
    }
    hips_world = 0.5 * (body_world["l_leg_pitch_link"] + body_world["r_leg_pitch_link"])
    neck_world = 0.5 * (body_world["l_arm_pitch_link"] + body_world["r_arm_pitch_link"])
    hips_local = _body_local_position(model, data, "base_link", hips_world)
    neck_local = _body_local_position(model, data, "waist_yaw_link", neck_world)

    waist_vertices = _mesh_vertices(source_mesh_dir / "waist_yaw_link.STL")
    head_local = np.array(
        [neck_local[0], neck_local[1], max(float(waist_vertices[:, 2].max()) + 0.015, neck_local[2] + 0.08)]
    )
    left_heel, left_toe, left_foot_report = _estimate_foot_markers(
        _mesh_vertices(source_mesh_dir / "l_ankle_roll_link.STL"), "left"
    )
    right_heel, right_toe, right_foot_report = _estimate_foot_markers(
        _mesh_vertices(source_mesh_dir / "r_ankle_roll_link.STL"), "right"
    )
    left_hand, left_hand_report = _estimate_hand_marker(
        _mesh_vertices(source_mesh_dir / "l_elbow_pitch_link.STL"), "left"
    )
    right_hand, right_hand_report = _estimate_hand_marker(
        _mesh_vertices(source_mesh_dir / "r_elbow_pitch_link.STL"), "right"
    )

    foot_lengths = [float(left_toe[0] - left_heel[0]), float(right_toe[0] - right_heel[0])]
    if abs(foot_lengths[0] - foot_lengths[1]) / max(foot_lengths) > 0.02:
        raise ValueError(f"左右脚 marker 长度差超过 2%: actual={foot_lengths}")
    if abs(float(left_heel[2] - right_heel[2])) > 0.005:
        raise ValueError(
            f"左右脚底高度差超过 5mm: left={left_heel[2]}, right={right_heel[2]}"
        )
    hand_distances = [float(np.linalg.norm(left_hand)), float(np.linalg.norm(right_hand))]
    if abs(hand_distances[0] - hand_distances[1]) / max(hand_distances) > 0.05:
        raise ValueError(f"左右手端距离差超过 5%: actual={hand_distances}")

    overrides = _load_overrides(overrides_path.expanduser().resolve())
    positions = {
        "hips_sphere": hips_local,
        "neck_sphere": neck_local,
        "head_sphere": head_local,
        "left_foot_end_link": left_heel,
        "left_toe_link": left_toe,
        "right_foot_end_link": right_heel,
        "right_toe_link": right_toe,
        "left_hand": left_hand,
        "right_hand": right_hand,
    }
    for name, value in overrides.items():
        if name not in positions:
            raise ValueError(f"未知 marker override: path={overrides_path}, field=markers.{name}")
        positions[name] = np.asarray(value, dtype=np.float64)

    parent_names = {
        "hips_sphere": "base_link",
        "neck_sphere": "waist_yaw_link",
        "head_sphere": "waist_yaw_link",
        "left_foot_end_link": "l_ankle_roll_link",
        "left_toe_link": "l_ankle_roll_link",
        "right_foot_end_link": "r_ankle_roll_link",
        "right_toe_link": "r_ankle_roll_link",
        "left_hand": "l_elbow_pitch_link",
        "right_hand": "r_elbow_pitch_link",
    }
    for name in BUMI3_MARKER_BODY_NAMES:
        _add_marker(_body_element(root, parent_names[name]), name, positions[name])

    ET.indent(tree, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    output_mjcf = output_dir / "mjcf" / "bumi3_retarget.xml"
    xml_status = _write_if_allowed(output_mjcf, xml_bytes, overwrite)

    report = {
        "source_dir": str(source_dir),
        "source_mjcf": str(source_mjcf),
        "source_mjcf_sha256": sha256_file(source_mjcf),
        "source_urdf": str(source_urdf),
        "source_urdf_sha256": sha256_file(source_urdf),
        "output_mjcf": str(output_mjcf),
        "output_mjcf_sha256": sha256_file(output_mjcf),
        "output_status": xml_status,
        "joint_count": len(urdf_limits),
        "joint_limit_changes": limit_changes,
        "mesh_files": copied_meshes,
        "markers": {
            name: {
                "parent": parent_names[name],
                "position_local": positions[name].tolist(),
                "overridden": name in overrides,
                "visible": name in VISIBLE_FOOT_MARKER_NAMES,
            }
            for name in BUMI3_MARKER_BODY_NAMES
        },
        "geometry_evidence": {
            "left_foot": left_foot_report,
            "right_foot": right_foot_report,
            "left_hand": left_hand_report,
            "right_hand": right_hand_report,
            "waist_mesh_z_max": float(waist_vertices[:, 2].max()),
        },
    }
    report_path = output_dir / "prepare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    if args.source_dir is None:
        raise ValueError("缺少 BUMI3 源目录：请传 --source-dir 或设置 BUMI_SOURCE_DIR")
    report = prepare_asset(args.source_dir, args.output_dir, args.overrides, args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
