#!/usr/bin/env bash
# 批量重定向本地四集合人工筛选的 10 条音乐 SMPL-X 动作。
#
# 本脚本确定性扫描 ``dataset/music_smpl_4set/<dataset>/*.npz``，要求文件总数恰好
# 为下载清单约定的 10 条，然后逐条调用经过联合验证的一键 BUMI3 流水线。每条
# 动作使用自身文件名作为 KEYPOINTS_NAME，产物统一写入 OUTPUT_DIR；任何一条失败
# 都立即停止并保留此前报告。TARGET_FPS 默认读取当前严格 G1 基线的 30 Hz，也可由
# 调用者显式覆盖。全部成功后调用多轨迹播放器的 ``--list-only``，再次核对播放器
# 实际能发现 10 条 CSV、各自 qpos shape、fps 和时长。脚本默认不打开图形窗口。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-dataset/music_smpl_4set}"
OUTPUT_DIR="${OUTPUT_DIR:-output_data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_FPS="${TARGET_FPS:-30}"
BUMI_SOURCE_DIR="${BUMI_SOURCE_DIR:-../legged_lab/source/NoetixRobot/NoetixRobot/assets/robots/bumi3}"
SMPL_MODEL_PATH="${SMPL_MODEL_PATH:-../GENMO/inputs/checkpoints/body_models}"

mapfile -t MOTIONS < <(find "${DATASET_ROOT}" -mindepth 2 -maxdepth 2 -type f -name '*.npz' | LC_ALL=C sort)
if [[ "${#MOTIONS[@]}" -ne 10 ]]; then
  echo "[失败] 四集合动作数量必须为 10，实际 ${#MOTIONS[@]}，目录 ${DATASET_ROOT}" >&2
  exit 2
fi

for index in "${!MOTIONS[@]}"; do
  motion="${MOTIONS[$index]}"
  motion_name="$(basename "${motion}" .npz)"
  echo "[批处理 $((index + 1))/10] ${motion}"
  BUMI_SOURCE_DIR="${BUMI_SOURCE_DIR}" \
  SMPL_MOTION_FILE="${motion}" \
  SMPL_MODEL_PATH="${SMPL_MODEL_PATH}" \
  MODEL_TYPE=smplx \
  TARGET_FPS="${TARGET_FPS}" \
  KEYPOINTS_NAME="${motion_name}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  RENDER_DEBUG=false \
  VISUALIZE=false \
  MAX_FRAMES=0 \
  PYTHON_BIN="${PYTHON_BIN}" \
  "${SCRIPT_DIR}/retarget_smpl_to_bumi3.sh"
done

"${PYTHON_BIN}" scripts/play_bumi3_trajectories.py \
  --motion-dir "${OUTPUT_DIR}/robot_motion" \
  --pattern '*_bumi3.csv' \
  --list-only

echo "[完成] 四集合 10 条 BUMI3 轨迹全部生成并通过逐条联合验证"
