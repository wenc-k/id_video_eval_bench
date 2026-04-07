CUDA_VISIBLE_DEVICES= python id_video_eval_system.py \
  --dino_model_path dino-vits16 \
  --dd_model_path raft_model/models/raft-things.pth \
  --amt_config_path amt_model/AMT-S.yaml \
  --amt_ckpt_path amt_model/amt-s.pth \
  --clip_cache_dir ip \
  --device cuda \
  --input_json  \
  --output_dir  \