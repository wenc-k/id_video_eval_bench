#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${SCRIPT_DIR}/../ckpt"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "${SCRIPT_DIR}/eval_system_v2.py" \
  --input_json "" \
  --output_dir "" \
  --ckpt_root "${CKPT_DIR}" \
  --aesthetic_script "${SCRIPT_DIR}/get_aesscore.py" \
  --facesim_script "${SCRIPT_DIR}/get_facesim.py" \
  --gmescore_script "${SCRIPT_DIR}/get_gmescore.py" \
  --motion_amplitude_script "${SCRIPT_DIR}/get_motion_amplitude.py" \
  --naturalscore_script "${SCRIPT_DIR}/get_naturalscore.py" \
  --aes_main_path "${CKPT_DIR}/aesthetic-model.pth" \
  --facesim_model_path "${CKPT_DIR}" \
  --skip_naturalscore True
