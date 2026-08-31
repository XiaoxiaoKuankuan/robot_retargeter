#!/usr/bin/env python3
"""四个高质量音乐舞蹈数据集到 BUMI3 的可恢复全量重定向入口。

本脚本面向 AIOZ-GDance、AIST++、CoMPAS3D 与 FineDance 的正式发布任务，
不是把十条演示脚本简单扩大循环次数。它先逐文件核对标准化 SMPL-X 输入的字段、
帧数、有限值、30 Hz、米制右手 Z-up 声明以及 Y-up 到 Z-up 的历史转换说明；随后
只执行一次 BUMI3 模型预检，再以受控并发调用单条重定向流水线。每个数据集使用
独立输出目录，因此相同 stem 不会互相覆盖；每条任务有独立日志，已经通过联合
验证且来源匹配的产物会安全续跑跳过。

正式完成时，脚本再次读取每条联合验证报告，要求 keypoint 明确记录“输入已是
Z-up、未二次旋转、输出仍是 Z-up”，并汇总源人体和目标机器人根倾角分布。发布
门禁检查每个数据集的根倾角中位数、P95 和异常动作占比；只有数量、逐条产物、
联合验证、坐标链路和分布门禁全部通过，``release_report.json`` 才会标为 passed。
单条舞蹈允许真实的地板动作或倒立，门禁针对整库分布，避免把合法 breakdance
误判为整库坐标错误，同时能够拒绝“所有动作约 90 度躺倒”的系统性错误。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


DATASET_EXPECTED_COUNTS = {
    "aioz_gdance": 1978,
    "aistpp": 963,
    "compas3d": 72,
    "finedance": 149,
}
EXPECTED_COORDINATE_SYSTEM = "right_handed_z_up_metric"
EXPECTED_SOURCE_COORDINATE_SYSTEM = "right_handed_y_up_metric"
EXPECTED_COORDINATE_TRANSFORM = (
    "rotate_global_root_and_translation_plus_90deg_about_x"
)
MAX_DATASET_MEDIAN_OF_MOTION_MEDIANS_DEG = 30.0
MAX_DATASET_P95_OF_MOTION_MEDIANS_DEG = 45.0
MAX_DATASET_HIGH_TILT_MOTION_RATE = 0.01
HIGH_TILT_MOTION_MEDIAN_DEG = 60.0


@dataclass(frozen=True)
class MotionTask:
    """保存一条源动作及其已验证的输入合同。"""

    dataset: str
    stem: str
    source_path: str
    frames: int
    fps: float
    coordinate_system: str
    source_coordinate_system: str
    coordinate_transform: str
    source_root_tilt_median_deg: float
    source_root_tilt_p95_deg: float
    source_sha256: str
    source_size_bytes: int


@dataclass(frozen=True)
class TaskResult:
    """保存一条重定向任务的终态，供进度与发布报告汇总。"""

    dataset: str
    stem: str
    source_path: str
    status: str
    elapsed_seconds: float
    log_path: str
    report_path: str
    message: str
    output_root_tilt_median_deg: float | None
    output_root_tilt_p95_deg: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="全量重定向四个 Z-up SMPL-X 音乐舞蹈数据集到 BUMI3"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--smpl-model-path", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发子进程数；服务器全量任务建议先 smoke 后按 CPU/内存设置",
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_EXPECTED_COUNTS),
        choices=list(DATASET_EXPECTED_COUNTS),
    )
    parser.add_argument(
        "--max-files-per-dataset",
        type=int,
        default=0,
        help="仅用于 smoke；大于 0 时每库只取排序后的前 N 条且不可发布",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="只执行全量输入坐标/字段/数量审计，不运行重定向",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不跳过已经通过全部坐标与联合验证门禁的同名产物",
    )
    return parser.parse_args()


def scalar_text(payload: np.lib.npyio.NpzFile, field: str) -> str:
    if field not in payload.files:
        raise ValueError(f"缺少字符串字段 {field}")
    values = np.asarray(payload[field]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"字段 {field} 必须是标量，实际 shape={values.shape}")
    return str(values[0]).strip()


def scalar_float(payload: np.lib.npyio.NpzFile, fields: tuple[str, ...]) -> float:
    for field in fields:
        if field in payload.files:
            values = np.asarray(payload[field]).reshape(-1)
            if values.size != 1:
                raise ValueError(f"字段 {field} 必须是标量，实际 shape={values.shape}")
            return float(values[0])
    raise ValueError(f"缺少帧率字段，候选={fields}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampled_root_tilt_degrees(
    root_orient: np.ndarray, maximum_samples: int = 500
) -> np.ndarray:
    """以 SMPL 局部 +Y 为身体向上轴，返回相对世界 +Z 的抽样倾角。"""
    values = np.asarray(root_orient, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError(f"root_orient 必须为非空 [T,3]，实际 {values.shape}")
    if values.shape[0] > maximum_samples:
        indices = np.linspace(
            0, values.shape[0] - 1, maximum_samples, dtype=np.int64
        )
        values = values[indices]
    body_up_world = Rotation.from_rotvec(values).apply(
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    )
    return np.rad2deg(
        np.arccos(np.clip(body_up_world[:, 2], -1.0, 1.0))
    )


def audit_motion(path: Path, dataset: str, expected_fps: float) -> MotionTask:
    """严格审计一条正式 Z-up SMPL-X 文件，不做任何自动坐标猜测。"""
    resolved = path.expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as payload:
        required = {"root_orient", "pose_body", "trans", "betas"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"缺少核心字段 {missing}")
        root_orient = np.asarray(payload["root_orient"], dtype=np.float64)
        pose_body = np.asarray(payload["pose_body"], dtype=np.float64)
        trans = np.asarray(payload["trans"], dtype=np.float64)
        frame_count = int(root_orient.shape[0]) if root_orient.ndim else 0
        if root_orient.shape != (frame_count, 3):
            raise ValueError(f"root_orient shape 错误: {root_orient.shape}")
        if pose_body.shape != (frame_count, 63):
            raise ValueError(f"pose_body shape 错误: {pose_body.shape}")
        if trans.shape != (frame_count, 3):
            raise ValueError(f"trans shape 错误: {trans.shape}")
        if frame_count < 2:
            raise ValueError(f"帧数不足: {frame_count}")
        for field, values in (
            ("root_orient", root_orient),
            ("pose_body", pose_body),
            ("trans", trans),
            ("betas", np.asarray(payload["betas"], dtype=np.float64)),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"字段 {field} 包含 NaN/Inf")
        fps = scalar_float(
            payload,
            ("mocap_frame_rate", "mocap_framerate", "fps", "frame_rate"),
        )
        coordinate_system = scalar_text(payload, "coordinate_system")
        source_coordinate_system = scalar_text(
            payload, "source_coordinate_system"
        )
        coordinate_transform = scalar_text(payload, "coordinate_transform")

    if not np.isfinite(fps) or abs(fps - expected_fps) > 1.0e-9:
        raise ValueError(f"源 fps 错误: expected={expected_fps}, actual={fps}")
    if coordinate_system != EXPECTED_COORDINATE_SYSTEM:
        raise ValueError(
            "正式输入必须明确为右手 Z-up 米制: "
            f"expected={EXPECTED_COORDINATE_SYSTEM}, actual={coordinate_system}"
        )
    if source_coordinate_system != EXPECTED_SOURCE_COORDINATE_SYSTEM:
        raise ValueError(
            "源坐标说明不匹配: "
            f"expected={EXPECTED_SOURCE_COORDINATE_SYSTEM}, "
            f"actual={source_coordinate_system}"
        )
    if coordinate_transform != EXPECTED_COORDINATE_TRANSFORM:
        raise ValueError(
            "坐标转换说明不匹配: "
            f"expected={EXPECTED_COORDINATE_TRANSFORM}, actual={coordinate_transform}"
        )
    tilts = sampled_root_tilt_degrees(root_orient)
    return MotionTask(
        dataset=dataset,
        stem=resolved.stem,
        source_path=str(resolved),
        frames=frame_count,
        fps=fps,
        coordinate_system=coordinate_system,
        source_coordinate_system=source_coordinate_system,
        coordinate_transform=coordinate_transform,
        source_root_tilt_median_deg=float(np.median(tilts)),
        source_root_tilt_p95_deg=float(np.percentile(tilts, 95.0)),
        source_sha256=sha256_file(resolved),
        source_size_bytes=resolved.stat().st_size,
    )


def distribution_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("根倾角汇总必须包含有限值")
    return {
        "motion_count": int(array.size),
        "median_of_motion_medians_deg": float(np.median(array)),
        "p95_of_motion_medians_deg": float(np.percentile(array, 95.0)),
        "maximum_motion_median_deg": float(np.max(array)),
        "high_tilt_motion_count": int(
            np.count_nonzero(array >= HIGH_TILT_MOTION_MEDIAN_DEG)
        ),
        "high_tilt_motion_rate": float(
            np.mean(array >= HIGH_TILT_MOTION_MEDIAN_DEG)
        ),
    }


def distribution_passes(summary: dict[str, float | int]) -> bool:
    return bool(
        float(summary["median_of_motion_medians_deg"])
        < MAX_DATASET_MEDIAN_OF_MOTION_MEDIANS_DEG
        and float(summary["p95_of_motion_medians_deg"])
        < MAX_DATASET_P95_OF_MOTION_MEDIANS_DEG
        and float(summary["high_tilt_motion_rate"])
        < MAX_DATASET_HIGH_TILT_MOTION_RATE
    )


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def output_paths(task: MotionTask, output_root: Path) -> dict[str, Path]:
    base = output_root / task.dataset
    return {
        "keypoints": base / "keypoints" / "bumi3" / f"{task.stem}_keypoints.pkl",
        "csv": base / "robot_motion" / f"{task.stem}_bumi3.csv",
        "metadata": base / "robot_motion" / f"{task.stem}_bumi3.meta.json",
        "npz": base / "mimic_npz" / "bumi3" / f"{task.stem}.npz",
        "report": base / "reports" / f"{task.stem}_bumi3.json",
        "log": output_root / "logs" / task.dataset / f"{task.stem}.log",
    }


def verified_artifact_state(
    task: MotionTask, output_root: Path, target_fps: float
) -> tuple[bool, str, float | None, float | None]:
    """只把完整通过、来源一致且没有二次旋转的产物视为可续跑。"""
    paths = output_paths(task, output_root)
    missing = [name for name, path in paths.items() if name != "log" and not path.is_file()]
    if missing:
        return False, f"缺少产物 {missing}", None, None
    if any(paths[name].stat().st_size <= 0 for name in ("keypoints", "csv", "metadata", "npz", "report")):
        return False, "存在空产物", None, None
    try:
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"JSON 读取失败: {error}", None, None
    if report.get("status") != "passed":
        return False, "联合验证未通过", None, None
    if Path(str(metadata.get("source_motion", ""))).resolve() != Path(task.source_path):
        return False, "metadata 来源动作不匹配", None, None
    if abs(float(metadata.get("fps", -1.0)) - target_fps) > 1.0e-9:
        return False, "metadata fps 不匹配", None, None
    motion_check = report.get("checks", {}).get("motion", {})
    contract = motion_check.get("coordinate_contract", {})
    expected_contract = {
        "source_coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        "requested_up_axis": "z",
        "y_up_to_z_up_conversion_applied": False,
        "output_coordinate_system": EXPECTED_COORDINATE_SYSTEM,
    }
    if contract != expected_contract:
        return False, f"坐标链路不匹配: {contract}", None, None
    root_tilt = motion_check.get("root_tilt_degrees", {})
    try:
        median = float(root_tilt["median"])
        p95 = float(root_tilt["p95"])
    except (KeyError, TypeError, ValueError):
        return False, "根倾角统计缺失", None, None
    if not np.isfinite(median) or not np.isfinite(p95):
        return False, "根倾角统计非有限值", None, None
    try:
        with paths["keypoints"].open("rb") as stream:
            keypoints = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        return False, f"keypoint 读取失败: {error}", None, None
    keypoint_contract = {
        "source_coordinate_system": keypoints.get("source_coordinate_system"),
        "requested_up_axis": keypoints.get("requested_up_axis"),
        "y_up_to_z_up_conversion_applied": keypoints.get(
            "y_up_to_z_up_conversion_applied"
        ),
        "output_coordinate_system": keypoints.get("output_coordinate_system"),
    }
    if keypoint_contract != expected_contract:
        return False, f"keypoint 坐标链路不匹配: {keypoint_contract}", None, None
    return True, "全部产物和坐标链路已通过", median, p95


def run_task(
    task: MotionTask,
    repository_root: Path,
    output_root: Path,
    python_bin: Path,
    smpl_model_path: Path,
    target_fps: float,
    resume: bool,
) -> TaskResult:
    paths = output_paths(task, output_root)
    if resume:
        valid, message, median, p95 = verified_artifact_state(
            task, output_root, target_fps
        )
        if valid:
            return TaskResult(
                task.dataset,
                task.stem,
                task.source_path,
                "skipped_verified",
                0.0,
                str(paths["log"]),
                str(paths["report"]),
                message,
                median,
                p95,
            )
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "SMPL_MOTION_FILE": task.source_path,
            "SMPL_MODEL_PATH": str(smpl_model_path),
            "MODEL_TYPE": "smplx",
            "TARGET_FPS": f"{target_fps:g}",
            "UP_AXIS": "z",
            "KEYPOINTS_NAME": task.stem,
            "OUTPUT_DIR": str(output_root / task.dataset),
            "RENDER_DEBUG": "false",
            "VISUALIZE": "false",
            "MAX_FRAMES": "0",
            "PYTHON_BIN": str(python_bin),
            "PREPARE_ASSET": "false",
            "RUN_PREFLIGHT": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    started = time.monotonic()
    with paths["log"].open("w", encoding="utf-8") as log_stream:
        log_stream.write(
            json.dumps(
                {
                    "dataset": task.dataset,
                    "stem": task.stem,
                    "source": task.source_path,
                    "up_axis": "z",
                    "target_fps": target_fps,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        log_stream.flush()
        completed = subprocess.run(
            [str(repository_root / "bash" / "retarget_smpl_to_bumi3.sh")],
            cwd=repository_root,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    valid, message, median, p95 = verified_artifact_state(
        task, output_root, target_fps
    )
    status = "completed" if completed.returncode == 0 and valid else "failed"
    if completed.returncode != 0:
        message = f"流水线退出码 {completed.returncode}; {message}"
    return TaskResult(
        task.dataset,
        task.stem,
        task.source_path,
        status,
        elapsed,
        str(paths["log"]),
        str(paths["report"]),
        message,
        median,
        p95,
    )


def git_value(repository_root: Path, arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    python_bin = args.python_bin.expanduser().resolve()
    smpl_model_path = args.smpl_model_path.expanduser().resolve()
    if args.workers <= 0:
        raise ValueError(f"workers 必须为正整数，实际 {args.workers}")
    if args.max_files_per_dataset < 0:
        raise ValueError("max-files-per-dataset 不得为负")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {dataset_root}")
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python 解释器不存在: {python_bin}")
    if not smpl_model_path.exists():
        raise FileNotFoundError(f"SMPL-X 模型路径不存在: {smpl_model_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    dependency_check = subprocess.run(
        [
            str(python_bin),
            "-c",
            "import mujoco,mink,numpy,scipy,smplx,torch,trimesh,yaml",
        ],
        cwd=repository_root,
        check=False,
    )
    if dependency_check.returncode != 0:
        raise RuntimeError("robot_retargeter 环境依赖检查失败")

    tasks: list[MotionTask] = []
    input_failures: list[dict[str, str]] = []
    actual_counts: dict[str, int] = {}
    for dataset in args.datasets:
        directory = dataset_root / dataset
        motions = sorted(directory.glob("*.npz")) if directory.is_dir() else []
        actual_counts[dataset] = len(motions)
        if args.max_files_per_dataset == 0:
            expected = DATASET_EXPECTED_COUNTS[dataset]
            if len(motions) != expected:
                input_failures.append(
                    {
                        "dataset": dataset,
                        "path": str(directory),
                        "error": f"数量错误: expected={expected}, actual={len(motions)}",
                    }
                )
        else:
            motions = motions[: args.max_files_per_dataset]
        stems = [path.stem for path in motions]
        if len(stems) != len(set(stems)):
            input_failures.append(
                {
                    "dataset": dataset,
                    "path": str(directory),
                    "error": "同一数据集存在重复 stem",
                }
            )
        for path in motions:
            try:
                tasks.append(audit_motion(path, dataset, args.target_fps))
            except Exception as error:  # noqa: BLE001 - 正式清单必须保留每条失败原因
                input_failures.append(
                    {
                        "dataset": dataset,
                        "path": str(path.resolve()),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    source_distribution = {
        dataset: distribution_summary(
            [
                task.source_root_tilt_median_deg
                for task in tasks
                if task.dataset == dataset
            ]
        )
        for dataset in args.datasets
        if any(task.dataset == dataset for task in tasks)
    }
    source_distribution_failures = [
        dataset
        for dataset, summary in source_distribution.items()
        if not distribution_passes(summary)
    ]
    input_manifest = {
        "schema": "robot_retargeter.bumi3_full_input_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "expected_counts": {
            dataset: DATASET_EXPECTED_COUNTS[dataset] for dataset in args.datasets
        },
        "actual_counts": actual_counts,
        "selected_task_count": len(tasks),
        "smoke_limit_per_dataset": args.max_files_per_dataset,
        "coordinate_contract": {
            "input": EXPECTED_COORDINATE_SYSTEM,
            "historical_source": EXPECTED_SOURCE_COORDINATE_SYSTEM,
            "historical_transform": EXPECTED_COORDINATE_TRANSFORM,
            "retarget_up_axis_argument": "z",
            "additional_rotation_in_retargeter": False,
        },
        "source_root_tilt_distribution": source_distribution,
        "source_root_tilt_failed_datasets": source_distribution_failures,
        "failures": input_failures,
        "motions": [asdict(task) for task in tasks],
    }
    write_json_atomic(output_root / "input_manifest.json", input_manifest)
    if input_failures or source_distribution_failures:
        print(
            json.dumps(
                {
                    "status": "input_audit_failed",
                    "failure_count": len(input_failures),
                    "tilt_failed_datasets": source_distribution_failures,
                    "manifest": str(output_root / "input_manifest.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    if args.audit_only:
        print(
            json.dumps(
                {
                    "status": "input_audit_passed",
                    "task_count": len(tasks),
                    "manifest": str(output_root / "input_manifest.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    preflight_report = output_root / "reports" / "bumi3_model_preflight.json"
    preflight_log = output_root / "logs" / "bumi3_model_preflight.log"
    preflight_report.parent.mkdir(parents=True, exist_ok=True)
    preflight_log.parent.mkdir(parents=True, exist_ok=True)
    with preflight_log.open("w", encoding="utf-8") as log_stream:
        preflight = subprocess.run(
            [
                str(python_bin),
                "scripts/validate_bumi3_retarget.py",
                "--config",
                "config/robot/bumi3.yaml",
                "--report",
                str(preflight_report),
            ],
            cwd=repository_root,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if preflight.returncode != 0:
        raise RuntimeError(f"BUMI3 模型预检失败，日志: {preflight_log}")

    started = time.monotonic()
    results: list[TaskResult] = []
    event_path = output_root / "progress.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_task,
                task,
                repository_root,
                output_root,
                python_bin,
                smpl_model_path,
                args.target_fps,
                not args.no_resume,
            ): task
            for task in tasks
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            event = {
                "completed": completed_count,
                "total": len(tasks),
                **asdict(result),
            }
            with event_path.open("a", encoding="utf-8") as event_stream:
                event_stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(
                f"[{completed_count}/{len(tasks)}] {result.dataset}/{result.stem} "
                f"{result.status} {result.elapsed_seconds:.1f}s",
                flush=True,
            )

    results.sort(key=lambda item: (item.dataset, item.stem))
    failed_results = [result for result in results if result.status == "failed"]
    output_distribution: dict[str, dict[str, float | int]] = {}
    output_distribution_failures: list[str] = []
    if not failed_results:
        for dataset in args.datasets:
            medians = [
                float(result.output_root_tilt_median_deg)
                for result in results
                if result.dataset == dataset
                and result.output_root_tilt_median_deg is not None
            ]
            output_distribution[dataset] = distribution_summary(medians)
            if not distribution_passes(output_distribution[dataset]):
                output_distribution_failures.append(dataset)

    smoke_mode = args.max_files_per_dataset > 0
    release_passed = not failed_results and not output_distribution_failures
    status = (
        "smoke_passed"
        if release_passed and smoke_mode
        else "passed" if release_passed else "failed"
    )
    release_report = {
        "schema": "robot_retargeter.bumi3_full_release.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "publishable": bool(release_passed and not smoke_mode),
        "repository": {
            "root": str(repository_root),
            "branch": git_value(repository_root, ["branch", "--show-current"]),
            "head": git_value(repository_root, ["rev-parse", "HEAD"]),
            "status_porcelain": git_value(repository_root, ["status", "--porcelain"]),
        },
        "runtime": {
            "python_bin": str(python_bin),
            "workers": args.workers,
            "target_fps": args.target_fps,
            "elapsed_seconds": time.monotonic() - started,
        },
        "contracts": {
            "config_sha256": sha256_file(repository_root / "config/robot/bumi3.yaml"),
            "mjcf_sha256": sha256_file(
                repository_root / "asset/robot/bumi3/mjcf/bumi3_retarget.xml"
            ),
            "input_manifest": str(output_root / "input_manifest.json"),
            "preflight_report": str(preflight_report),
        },
        "counts": {
            "selected": len(tasks),
            "completed": sum(result.status == "completed" for result in results),
            "skipped_verified": sum(
                result.status == "skipped_verified" for result in results
            ),
            "failed": len(failed_results),
            "per_dataset": {
                dataset: sum(result.dataset == dataset for result in results)
                for dataset in args.datasets
            },
        },
        "root_tilt_gate": {
            "thresholds": {
                "maximum_median_of_motion_medians_deg": (
                    MAX_DATASET_MEDIAN_OF_MOTION_MEDIANS_DEG
                ),
                "maximum_p95_of_motion_medians_deg": (
                    MAX_DATASET_P95_OF_MOTION_MEDIANS_DEG
                ),
                "high_tilt_motion_median_deg": HIGH_TILT_MOTION_MEDIAN_DEG,
                "maximum_high_tilt_motion_rate": MAX_DATASET_HIGH_TILT_MOTION_RATE,
            },
            "source": source_distribution,
            "output": output_distribution,
            "failed_output_datasets": output_distribution_failures,
        },
        "failures": [asdict(result) for result in failed_results],
        "results": [asdict(result) for result in results],
    }
    write_json_atomic(output_root / "release_report.json", release_report)
    print(
        json.dumps(
            {
                "status": status,
                "publishable": release_report["publishable"],
                "counts": release_report["counts"],
                "report": str(output_root / "release_report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if release_passed else 1


if __name__ == "__main__":
    sys.exit(main())
