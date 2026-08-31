#!/usr/bin/env python3
"""在单个 MuJoCo 窗口中播放并即时切换多条机器人重定向轨迹。

脚本默认扫描 ``output_data/robot_motion`` 下的 ``*_bumi3.csv``，也可显式传入
任意数量的 CSV。它复用命令行配置中的 MJCF，加载时验证每条轨迹的 qpos
列数和有限值，并把 CSV 根四元数从 ``xyzw`` 转回 MuJoCo 的 ``wxyz``。播放
期间空格暂停/继续，左右方向键或 P/N 切换上一条/下一条，R 从当前轨迹首帧
重播，数字 1～9 与 0 可直接选择第 1～10 条轨迹；切换时同一 viewer 保持
打开。每条 CSV 优先读取同名 ``.meta.json`` 的实际 fps，也可以用命令行统一
覆盖。播放器会在启动、切换和重播时将自由相机对准当前轨迹的根节点，并
提高 MJCF 默认头灯亮度，避免带世界平移的轨迹落在静态模型相机范围之外而
显示为纯黑窗口。若机器人 MJCF 没有平面，脚本还会按仓库原多机器人播放器
的参数补充渐变天空盒、灰色网格地板和方向光；已有地板的模型不会重复添加。
``--list-only`` 可在无图形环境中完成发现、顺序和 shape 预检。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from bumi3_common import load_yaml, resolve_config_path
from export_bumi3_mimic_npz import load_csv_qpos


@dataclass(frozen=True)
class MotionClip:
    """一条已经通过基本格式检查的 BUMI3 qpos 轨迹。"""

    path: Path
    qpos: np.ndarray
    fps: float


_COMMAND: str | int | None = None
_PAUSED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="播放并切换 BUMI3 MuJoCo 轨迹")
    parser.add_argument("motions", nargs="*", type=Path, help="显式 CSV 列表")
    parser.add_argument(
        "--motion-dir",
        type=Path,
        default=Path("output_data/robot_motion"),
        help="未显式给 CSV 时扫描的目录",
    )
    parser.add_argument("--pattern", default="*_bumi3.csv", help="目录扫描 glob")
    parser.add_argument(
        "--config", type=Path, default=Path("config/robot/bumi3.yaml"), help="BUMI3 配置"
    )
    parser.add_argument("--fps", type=float, default=0.0, help="大于 0 时覆盖所有轨迹 fps")
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=0.0,
        help="相机距离；不大于 0 时根据模型尺度自动计算",
    )
    parser.add_argument("--no-loop", action="store_true", help="末帧暂停而非循环")
    parser.add_argument("--list-only", action="store_true", help="只列出并验证轨迹，不开窗口")
    return parser.parse_args()


def metadata_path_for_csv(csv_path: Path) -> Path:
    """返回仓库约定的 ``<stem>.meta.json`` 路径。"""
    return csv_path.with_suffix(".meta.json")


def load_clip(csv_path: Path, model: mujoco.MjModel, fps_override: float) -> MotionClip:
    """加载单条 CSV，并从元数据读取实际 fps。"""
    resolved = csv_path.expanduser().resolve()
    qpos = load_csv_qpos(resolved, model)
    if fps_override > 0.0:
        fps = float(fps_override)
    else:
        metadata_path = metadata_path_for_csv(resolved)
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"轨迹缺少同名元数据且未指定 --fps: path={metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fps = float(metadata.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"轨迹 fps 必须为正有限值: path={resolved}, actual={fps}")
    return MotionClip(path=resolved, qpos=qpos, fps=fps)


def discover_motion_paths(explicit: list[Path], directory: Path, pattern: str) -> list[Path]:
    """确定性返回显式列表或目录 glob 的轨迹路径。"""
    paths = explicit if explicit else sorted(directory.expanduser().resolve().glob(pattern))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    if not unique:
        raise FileNotFoundError(
            f"未找到 BUMI3 CSV: directory={directory.expanduser().resolve()}, pattern={pattern}"
        )
    return unique


def key_callback(keycode: int) -> None:
    """将 viewer 键盘事件转成主循环安全消费的单一命令。"""
    global _COMMAND, _PAUSED
    if keycode == glfw.KEY_SPACE:
        _PAUSED = not _PAUSED
    elif keycode in (glfw.KEY_RIGHT, glfw.KEY_N):
        _COMMAND = "next"
    elif keycode in (glfw.KEY_LEFT, glfw.KEY_P):
        _COMMAND = "previous"
    elif keycode == glfw.KEY_R:
        _COMMAND = "restart"
    elif glfw.KEY_1 <= keycode <= glfw.KEY_9:
        _COMMAND = int(keycode - glfw.KEY_1)
    elif keycode == glfw.KEY_0:
        _COMMAND = 9


def print_clip(index: int, clips: list[MotionClip]) -> None:
    """打印当前轨迹名称、序号、帧数、时长和帧率。"""
    clip = clips[index]
    print(
        f"[MuJoCo] 轨迹 {index + 1}/{len(clips)}: {clip.path.name}, "
        f"frames={clip.qpos.shape[0]}, fps={clip.fps:g}, "
        f"duration={clip.qpos.shape[0] / clip.fps:.2f}s"
    )


def brighten_model(model: mujoco.MjModel) -> None:
    """提高默认头灯亮度，使没有自带场景灯的机器人 MJCF 仍清晰可见。"""
    model.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
    model.vis.headlight.diffuse[:] = (0.70, 0.70, 0.70)
    model.vis.headlight.specular[:] = (0.25, 0.25, 0.25)


def build_viewer_model(xml_path: Path) -> mujoco.MjModel:
    """加载机器人 MJCF，并在缺少平面时补充原播放器同款网格场景。"""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    if any(geom.type == mujoco.mjtGeom.mjGEOM_PLANE for geom in spec.geoms):
        return spec.compile()

    if not any(
        texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX
        for texture in spec.textures
    ):
        spec.add_texture(
            name="viewer_skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0.0, 0.0, 0.0],
            width=512,
            height=512,
        )

    grid_resolution = 256
    line_width = 3
    grid_rgba = np.zeros((grid_resolution, grid_resolution, 4), dtype=np.uint8)
    grid_rgba[:, :, :3] = (100, 100, 100)
    grid_rgba[:, :, 3] = 255
    grid_rgba[:line_width, :, :3] = (50, 50, 50)
    grid_rgba[-line_width:, :, :3] = (50, 50, 50)
    grid_rgba[:, :line_width, :3] = (50, 50, 50)
    grid_rgba[:, -line_width:, :3] = (50, 50, 50)

    grid_texture = spec.add_texture(
        name="viewer_groundplane_texture",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=grid_resolution,
        height=grid_resolution,
        nchannel=4,
    )
    grid_texture.data = grid_rgba.reshape(-1).tobytes()
    spec.add_material(
        name="viewer_groundplane_material",
        textures=["", "viewer_groundplane_texture"],
        texrepeat=[10, 10],
        texuniform=True,
        reflectance=0.0,
    )
    spec.worldbody.add_light(
        name="viewer_ground_light",
        pos=[0.0, 0.0, 20.0],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.7, 0.7, 0.7],
        specular=[0.3, 0.3, 0.3],
    )
    spec.worldbody.add_geom(
        name="viewer_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        material="viewer_groundplane_material",
        pos=[0.0, 0.0, 0.0],
        contype=0,
        conaffinity=0,
    )
    return spec.compile()


def focus_camera(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    clip: MotionClip,
    frame_index: int,
    distance_override: float,
) -> None:
    """把自由相机对准当前帧根节点，而不是 MJCF 的静态模型中心。"""
    root_position = clip.qpos[frame_index, :3]
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = root_position
    viewer.cam.distance = (
        float(distance_override)
        if distance_override > 0.0
        else max(2.5, 2.2 * float(model.stat.extent))
    )
    viewer.cam.azimuth = -135.0
    viewer.cam.elevation = -20.0


def play(
    model: mujoco.MjModel,
    clips: list[MotionClip],
    loop: bool,
    camera_distance: float,
) -> None:
    """运行单窗口播放循环并响应即时轨迹切换。"""
    global _COMMAND, _PAUSED
    data = mujoco.MjData(model)
    clip_index = 0
    frame_index = 0
    _COMMAND = None
    _PAUSED = False
    brighten_model(model)
    data.qpos[:] = clips[clip_index].qpos[frame_index]
    mujoco.mj_forward(model, data)
    print("控制：Space 暂停/继续；←/P 上一条；→/N 下一条；R 重播；1～9/0 直选")
    print_clip(clip_index, clips)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        focus_camera(viewer, model, clips[clip_index], frame_index, camera_distance)
        viewer.sync()
        while viewer.is_running():
            command = _COMMAND
            _COMMAND = None
            refocus = False
            if command == "next":
                clip_index = (clip_index + 1) % len(clips)
                frame_index = 0
                refocus = True
                print_clip(clip_index, clips)
            elif command == "previous":
                clip_index = (clip_index - 1) % len(clips)
                frame_index = 0
                refocus = True
                print_clip(clip_index, clips)
            elif command == "restart":
                frame_index = 0
                refocus = True
                print_clip(clip_index, clips)
            elif isinstance(command, int) and command < len(clips):
                clip_index = command
                frame_index = 0
                refocus = True
                print_clip(clip_index, clips)

            clip = clips[clip_index]
            start_time = time.perf_counter()
            data.qpos[:] = clip.qpos[frame_index]
            mujoco.mj_forward(model, data)
            if refocus:
                focus_camera(viewer, model, clip, frame_index, camera_distance)
            viewer.sync()
            if not _PAUSED:
                frame_index += 1
                if frame_index >= clip.qpos.shape[0]:
                    if loop:
                        frame_index = 0
                    else:
                        frame_index = clip.qpos.shape[0] - 1
                        _PAUSED = True
            elapsed = time.perf_counter() - start_time
            time.sleep(max(0.0, 1.0 / clip.fps - elapsed))


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    xml_path = resolve_config_path(config_path, str(config["robot_xml_path"]))
    model = build_viewer_model(xml_path)
    paths = discover_motion_paths(args.motions, args.motion_dir, args.pattern)
    clips = [load_clip(path, model, args.fps) for path in paths]
    for index in range(len(clips)):
        print_clip(index, clips)
    if not args.list_only:
        play(
            model,
            clips,
            loop=not args.no_loop,
            camera_distance=args.camera_distance,
        )


if __name__ == "__main__":
    main()
