python preprocess_video.py \
    --video_path /data/ziqi/Repos/HY-WorldPlay-New/outputs/butter_all_motions/d7dleft4/qwen_tod7dleft4.mp4  \
    --pose_string "left-7, dleft-4" \
    --prompt "" \
    --output_dir training_data/butter_qwen_d7dleft4/ \
    --model_path /data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038 \
    --height 432 \
    --width 768