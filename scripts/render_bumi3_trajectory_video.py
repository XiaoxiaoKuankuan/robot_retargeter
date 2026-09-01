#!/usr/bin/env python3
"""把 BUMI3 qpos CSV 离屏渲染成可复核的 MP4 视频。

脚本读取仓库标准的 CSV 与同名 metadata，从元数据获得真实帧率，并复用交互播放器
完全相同的 MuJoCo 模型、灯光和网格地板补全逻辑。每帧通过 MuJoCo 原生 Renderer
生成 RGB 图像，再用 ffmpeg 流式编码为 H.264/yuv420p；不会先在磁盘堆积数千张临时
PNG。相机水平跟随当前根位置，距离、方位角和俯仰角在整段中保持不变，因此既能看清
机器人，也不会因源动作世界平移离开固定相机而得到黑屏。默认拒绝覆盖已有视频，
只有显式传入 ``--overwrite`` 才允许替换，便于保留优化前后的 A/B 证据。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from tqdm import tqdm

from bumi3_common import load_yaml, resolve_config_path
from play_bumi3_trajectories import build_viewer_model, brighten_model, load_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离屏渲染 BUMI3 轨迹视频")
    parser.add_argument("csv", type=Path, help="BUMI3 qpos CSV")
    parser.add_argument("--output", type=Path, required=True, help="输出 MP4")
    parser.add_argument(
        "--config", type=Path, default=Path("config/robot/bumi3.yaml"), help="机器人配置"
    )
    parser.add_argument("--width", type=int, default=1280, help="视频宽度")
    parser.add_argument("--height", type=int, default=720, help="视频高度")
    parser.add_argument("--camera-distance", type=float, default=2.5, help="相机距离")
    parser.add_argument("--azimuth", type=float, default=-135.0, help="相机方位角（度）")
    parser.add_argument("--elevation", type=float, default=-20.0, help="相机俯仰角（度）")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF，越小质量越高")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出")
    return parser.parse_args()


def render_video(args: argparse.Namespace) -> None:
    """加载并逐帧渲染轨迹，编码器失败时保留明确的非零退出码。"""
    if args.width <= 0 or args.height <= 0:
        raise ValueError("视频宽高必须为正")
    if args.camera_distance <= 0.0:
        raise ValueError("camera-distance 必须为正")
    if not 0 <= args.crf <= 51:
        raise ValueError("crf 必须在 [0,51]")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("PATH 中找不到 ffmpeg")

    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在；如需覆盖请加 --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    xml_path = resolve_config_path(config_path, str(config["robot_xml_path"]))
    model = build_viewer_model(xml_path)
    brighten_model(model)
    # MuJoCo 默认离屏缓冲仅 640x480；这里按请求分辨率扩大模型的 framebuffer，
    # 不修改仓库 MJCF，也不影响交互播放器和动力学资产 SHA。
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)
    clip = load_clip(args.csv, model, fps_override=0.0)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(args.camera_distance)
    camera.azimuth = float(args.azimuth)
    camera.elevation = float(args.elevation)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if args.overwrite else "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{args.width}x{args.height}",
        "-framerate",
        f"{clip.fps:.12g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("无法打开 ffmpeg stdin")
    try:
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            for qpos in tqdm(clip.qpos, desc="Rendering", unit="frame"):
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                camera.lookat[:] = np.asarray(qpos[:3], dtype=np.float64)
                renderer.update_scene(data, camera=camera)
                frame = np.ascontiguousarray(renderer.render(), dtype=np.uint8)
                process.stdin.write(frame.tobytes())
    except BaseException:
        process.stdin.close()
        process.terminate()
        process.wait()
        raise
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg 编码失败: exit_code={return_code}")
    print(
        f"[完成] {output} frames={clip.qpos.shape[0]} fps={clip.fps:g} "
        f"duration={clip.qpos.shape[0] / clip.fps:.3f}s"
    )


def main() -> None:
    render_video(parse_args())


if __name__ == "__main__":
    main()
