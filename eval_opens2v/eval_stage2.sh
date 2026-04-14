#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) 原始统一输入 case json
INPUT_JSON="../phantom_generated_videos/phantom_results.json"

# 2) stage1 已有的 case_results_v2.json（包含除 qalign motion_smoothness 外的其余指标）
EXISTING_CASE_JSON="metric_phantom_v2/case_results_v2.json"

# 4) 复用 stage1 的 _v2_work/input_videos 目录
REUSE_INPUT_VIDEO_DIR="metric_phantom_v2/_v2_work/input_videos" # TODO 更换 metric_phantom_v2 即可 

# 3) 新 merge 结果输出目录
OUTPUT_DIR="phantom_results_done"

# 5) qalign / one-align 模型路径
MOTION_SMOOTHNESS_MODEL_PATH="q-future/one-align"

CUDA_VISIBLE_DEVICES=2 python "${SCRIPT_DIR}/eval_stage2.py" \
  --input_json "${INPUT_JSON}" \
  --existing_case_json "${EXISTING_CASE_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --motion_smoothness_model_path "${MOTION_SMOOTHNESS_MODEL_PATH}" \
  --reuse_input_video_dir "${REUSE_INPUT_VIDEO_DIR}"
