#!/usr/bin/env bash
# BUMI3 的 SMPL/SMPL-X 到 IsaacLab Mimic 一键重定向入口。
#
# 本脚本依次准备本地权威 BUMI3 资产、执行模型预检、将人体动作真实重采样为
# TARGET_FPS 指定频率的 BUMI3 尺寸关键点、运行仓库既有的单阶段 Mink IK、保存 qpos CSV 与
# 审计元数据、导出 21 关节/22 body Mimic NPZ，并运行最终联合验证。每一步都
# 打印编号与完整命令，任何命令失败都会立即退出并指出失败步骤。默认不打开
# 图形窗口；VISUALIZE=true 时用多轨迹播放器打开本次结果。所有路径均可通过
# 下列环境变量覆盖，脚本本身不绑定用户机器的绝对路径。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BUMI_SOURCE_DIR="${BUMI_SOURCE_DIR:-../legged_lab/source/NoetixRobot/NoetixRobot/assets/robots/bumi3}"
SMPL_MOTION_FILE="${SMPL_MOTION_FILE:-}"
SMPL_MODEL_PATH="${SMPL_MODEL_PATH:-../GENMO/inputs/checkpoints/body_models}"
MODEL_TYPE="${MODEL_TYPE:-auto}"
TARGET_FPS="${TARGET_FPS:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-output_data}"
RENDER_DEBUG="${RENDER_DEBUG:-false}"
VISUALIZE="${VISUALIZE:-false}"
MAX_FRAMES="${MAX_FRAMES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKELETON_CONFIG="${SKELETON_CONFIG:-config/skeleton/skeleton.yaml}"

if [[ -z "${SMPL_MOTION_FILE}" ]]; then
  echo "[失败] 必须设置 SMPL_MOTION_FILE=/path/to/motion.npz" >&2
  exit 2
fi
if [[ ! -f "${SMPL_MOTION_FILE}" ]]; then
  echo "[失败] SMPL 动作不存在: ${SMPL_MOTION_FILE}" >&2
  exit 2
fi
if [[ ! -d "${BUMI_SOURCE_DIR}" ]]; then
  echo "[失败] BUMI3 源资产目录不存在: ${BUMI_SOURCE_DIR}" >&2
  exit 2
fi
if [[ ! -e "${SMPL_MODEL_PATH}" ]]; then
  echo "[失败] SMPL 模型路径不存在: ${SMPL_MODEL_PATH}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c "import mujoco, mink, smplx, torch, yaml, scipy, trimesh"; then
  echo "[失败] PYTHON_BIN 缺少运行依赖；请激活 robot_retargeter 环境或设置正确解释器" >&2
  exit 2
fi
if [[ "${RENDER_DEBUG}" != "true" && "${RENDER_DEBUG}" != "false" ]]; then
  echo "[失败] RENDER_DEBUG 只能是 true/false，实际 ${RENDER_DEBUG}" >&2
  exit 2
fi
if [[ "${VISUALIZE}" != "true" && "${VISUALIZE}" != "false" ]]; then
  echo "[失败] VISUALIZE 只能是 true/false，实际 ${VISUALIZE}" >&2
  exit 2
fi

KEYPOINTS_NAME="${KEYPOINTS_NAME:-$(basename "${SMPL_MOTION_FILE}" .npz)}"
KEYPOINTS_PATH="${OUTPUT_DIR}/keypoints/bumi3/${KEYPOINTS_NAME}_keypoints.pkl"
CSV_PATH="${OUTPUT_DIR}/robot_motion/${KEYPOINTS_NAME}_bumi3.csv"
METADATA_PATH="${OUTPUT_DIR}/robot_motion/${KEYPOINTS_NAME}_bumi3.meta.json"
MIMIC_PATH="${OUTPUT_DIR}/mimic_npz/bumi3/${KEYPOINTS_NAME}.npz"
PREFLIGHT_REPORT="${OUTPUT_DIR}/reports/bumi3_model_preflight.json"
FINAL_REPORT="${OUTPUT_DIR}/reports/${KEYPOINTS_NAME}_bumi3.json"
CURRENT_STEP="初始化"

on_error() {
  local exit_code=$?
  echo "[失败] 步骤 '${CURRENT_STEP}'，退出码 ${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR

run_step() {
  local step_name="$1"
  shift
  CURRENT_STEP="${step_name}"
  echo "[${step_name}]"
  printf '  命令:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run_step "1/8 准备 BUMI3 资产" \
  "${PYTHON_BIN}" scripts/prepare_bumi3_asset.py \
  --source-dir "${BUMI_SOURCE_DIR}" \
  --output-dir asset/robot/bumi3 \
  --overrides config/robot/bumi3_marker_overrides.yaml

run_step "2/8 BUMI3 模型预检" \
  "${PYTHON_BIN}" scripts/validate_bumi3_retarget.py \
  --config config/robot/bumi3.yaml \
  --report "${PREFLIGHT_REPORT}"

SMPL_ARGS=(
  --no-viewer
  --motion_file "${SMPL_MOTION_FILE}"
  --smpl-model-path "${SMPL_MODEL_PATH}"
  --model-type "${MODEL_TYPE}"
  --target-fps "${TARGET_FPS}"
  --robot-config config/robot/bumi3.yaml
  --skeleton-config "${SKELETON_CONFIG}"
  --keypoints-name "${KEYPOINTS_NAME}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ "${MAX_FRAMES}" != "0" ]]; then
  SMPL_ARGS+=(--max-frames "${MAX_FRAMES}")
fi
run_step "3/8 生成 ${TARGET_FPS} Hz BUMI3 关键点" "${PYTHON_BIN}" scripts/smpl_replay.py "${SMPL_ARGS[@]}"

RETARGET_RENDER_ARG="--no-render-debug"
if [[ "${RENDER_DEBUG}" == "true" ]]; then
  RETARGET_RENDER_ARG="--render-debug"
fi
run_step "4/8 单阶段 Mink IK" \
  "${PYTHON_BIN}" scripts/robot_retarget.py \
  --config config/robot/bumi3.yaml \
  --keypoints-name "${KEYPOINTS_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  "${RETARGET_RENDER_ARG}"

CURRENT_STEP="5/8 核对 CSV 与元数据"
echo "[${CURRENT_STEP}]"
test -s "${CSV_PATH}"
test -s "${METADATA_PATH}"
echo "  CSV: ${CSV_PATH}"
echo "  JSON: ${METADATA_PATH}"

run_step "6/8 导出 IsaacLab Mimic NPZ" \
  "${PYTHON_BIN}" scripts/export_bumi3_mimic_npz.py \
  --csv "${CSV_PATH}" \
  --metadata "${METADATA_PATH}" \
  --config config/robot/bumi3.yaml \
  --output "${MIMIC_PATH}"

run_step "7/8 最终联合验证" \
  "${PYTHON_BIN}" scripts/validate_bumi3_retarget.py \
  --config config/robot/bumi3.yaml \
  --keypoints "${KEYPOINTS_PATH}" \
  --csv "${CSV_PATH}" \
  --metadata "${METADATA_PATH}" \
  --npz "${MIMIC_PATH}" \
  --report "${FINAL_REPORT}"

CURRENT_STEP="8/8 可选可视化"
echo "[${CURRENT_STEP}]"
if [[ "${VISUALIZE}" == "true" ]]; then
  "${PYTHON_BIN}" scripts/play_bumi3_trajectories.py "${CSV_PATH}" --config config/robot/bumi3.yaml
else
  echo "  已跳过；设置 VISUALIZE=true 可打开 MuJoCo 播放器"
fi

trap - ERR
echo "[完成] BUMI3 重定向、Mimic 导出与验证全部结束"
echo "  验证报告: ${FINAL_REPORT}"
