#!/bin/bash

#Ryan: Please run this from the PARENT of the scripts directory!

python ./source/datasets/youtube/DaS/Vanilla/make_columns.py
#That generates the following files:
#    ./source/datasets/youtube/DaS/Vanilla/prompt.txt
#    ./source/datasets/youtube/DaS/Vanilla/videos.txt
#    ./source/datasets/youtube/DaS/Vanilla/trackings.txt

set -x

export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30

GPU_IDS="0,1,2,3,4,5,6,7"
NUM_PROCESSES=8
PORT=29500
# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("1e-4")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("100000")
WARMUP_STEPS=100
CHECKPOINT_STEPS=500
CHECKPOINT_STEPS=5 #TINY MODE FOR TESTING: Use with source/checkpoint_pruner.py so you don't run out of harddrive space.
TRAIN_BATCH_SIZE=2 

RUN_NAME="CounterChans2500"

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_2.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.

# training dataset parameters
DATA_ROOT="/"
# MODEL_PATH="./ckpts/CogVideoX-5b-I2V"
MODEL_PATH="./diffusion_shader_model"
OUTPUT_PATH="./ckpts/your_ckpt_path"
CAPTION_COLUMN="$HOME/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/Vanilla/prompt.txt"
VIDEO_COLUMN="$HOME/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/Vanilla/videos.txt"
TRACKING_COLUMN="$HOME/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/Vanilla/trackings.txt"
COUNTER_TRACKING_COLUMN="$HOME/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/Vanilla/counter_trackings.txt"
COUNTER_VIDEO_COLUMN="$HOME/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/Vanilla/counter_videos.txt"

# validation parameters
TRACKING_MAP_PATH="$HOME/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/video.mp4__DiffusionAsShaderCondition/tracking_video.mp4"
COUNTER_TRACKING_MAP_PATH="$HOME/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/firstLastInterp_Jack2000.mp4__DiffusionAsShaderCondition/tracking_video.mp4"
COUNTER_VIDEO_MAP_PATH="$HOME/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/firstLastInterp_Jack2000.mp4"
VALIDATION_PROMPT="A soccer player from Hertha BSC is in the field with the ball while an opposing player is running towards him."
VALIDATION_IMAGES="$HOME/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/video_firstFrame.png"

# Launch experiments with different hyperparameters
for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="${OUTPUT_PATH}/${RUN_NAME}${steps}__optimizer_${optimizer}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS --num_processes $NUM_PROCESSES --main_process_port $PORT training/cogvideox_image_to_video_sft.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --tracking_column $TRACKING_COLUMN \
          --counter_tracking_column $COUNTER_TRACKING_COLUMN \
          --counter_video_column $COUNTER_VIDEO_COLUMN \
          --tracking_map_path $TRACKING_MAP_PATH \
          --counter_tracking_map_path $COUNTER_TRACKING_MAP_PATH \
          --counter_video_map_path $COUNTER_VIDEO_MAP_PATH \
          --num_tracking_blocks 18 \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 49 \
          --dataloader_num_workers 8 \
          --pin_memory \
          --validation_prompt \"$VALIDATION_PROMPT\" \
          --validation_images \"$VALIDATION_IMAGES\" \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 42 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size $TRAIN_BATCH_SIZE \
          --max_train_steps $steps \
          --checkpointing_steps $CHECKPOINT_STEPS \
          --gradient_accumulation_steps 4 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps $WARMUP_STEPS \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --noised_image_dropout 0.05 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
          --resume_from_checkpoint \"latest\" \
          --nccl_timeout 1800"
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
