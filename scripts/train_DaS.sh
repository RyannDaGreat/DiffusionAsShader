#!/bin/bash

'
Ryan: Please run this from the PARENT of the scripts directory!
!
# cd ~/CleanCode/Github/DiffusionAsShader
cd ~/CleanCode/Github/DaS_Trees/gauss_blobs

bash ~/CleanCode/Management/sync_projects.bash

INIT_CHECKPOINT_PATH="$HOME/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans_RandomSpeed_WithDropout_2500_10000000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-44200"
INIT_CHECKPOINT_PATH="$HOME/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans_FIXED_DATASET_BetterAug_WithDropout_50kSamp_T2V_from_scratch_10000000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-5500"
python ~/CleanCode/Management/syncutil.py sync_checkpoint_folder $INIT_CHECKPOINT_PATH

#Copy things from the original folder - these arent tracked by git
cp -al $HOME/CleanCode/Github/DiffusionAsShader/diffusion_shader_model diffusion_as_shader_model

TRAIN_FROM_DIR=diffusion_shader_model_start
rm -rf $TRAIN_FROM_DIR
cp -al ~/CleanCode/Github/DiffusionAsShader/diffusion_shader_model $TRAIN_FROM_DIR
rm -rf $TRAIN_FROM_DIR/transformer
cp -al $INIT_CHECKPOINT_PATH/transformer $TRAIN_FROM_DIR/transformer

bash scripts/train_DaS.sh



TO SYNC CHECKPOINTS:

while True:
    try:
        sys.path.append('/home/jupyter/CleanCode/Management')
        import syncutil
        most_recent_checkpoint = get_all_folders(get_all_folders('/home/jupyter/CleanCode/Github/DaS_Trees/gauss_blobs',sort_by='date')[-1],sort_by='date')[-1]
        string_to_clipboard(most_recent_checkpoint)
        syncutil.sync_checkpoint_folder(most_recent_checkpoint)
        print(fansi_highlight_path(most_recent_checkpoint))
        sleep(60*5)
    except IndexError:
        pass
    

TO RUN TESTS:
    tmux setw synchronize-panes
    PYM source.run_pipeline


'

##########################
##### EXTRA SETTINGS #####
##########################

export T2V_TRANSFORMER_CHECKPOINT="/home/jupyter/CleanCode/Huggingface/CogVideoX-5b/transformer"

#####################



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
MAX_TRAIN_STEPS=("10000000")
WARMUP_STEPS=100
CHECKPOINT_STEPS=500
# CHECKPOINT_STEPS=100 #TINY MODE FOR TESTING: Use with source/checkpoint_pruner.py so you don't run out of harddrive space.
TRAIN_BATCH_SIZE=2 

#Realized mistake - all last frames of counter and actual were same. We will train next from this.
#Was trained on 2500 samples. Used channel concatenation.
# RUN_NAME="CounterChans2500" #LAST TRAINED CHECKPOINT: /home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans2500100000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-5000

#Picking up from CounterChans2500, we add speed augmentations to each one - crudely, resulting in duplicate frames. Crucially however it makes it so first/last frame is now always the same.
RUN_NAME="CounterChans_RandomSpeed_10000"
RUN_NAME="CounterChans_RandomSpeed_2500_" #might have corrupted shrunken 720p between 2500 and 5000 vids...
RUN_NAME="CounterChans_RandomSpeed_WithDropout_2500_" #might have corrupted shrunken 720p between 2500 and 5000 vids...
RUN_NAME="CounterChans_BetterAug_WithDropout_50kSamp_T2V_" #

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_2.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.

# training dataset parameters
DATA_ROOT="/"
# MODEL_PATH="./ckpts/CogVideoX-5b-I2V"
# MODEL_PATH="./diffusion_shader_model"
# MODEL_PATH="./diffusion_shader_model_3666Start"


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

#VARIANTS TREE
  #DOING IT FROM SCRATCH 
  MODEL_PATH="/home/jupyter/CleanCode/Huggingface/CogVideoX-5b"
  RUN_NAME="CounterChans_FIXED_DATASET_BetterAug_WithDropout_50kSamp_T2V_from_scratch_"
    
    #CONTINUING AFTER DATASET FIX AUGMENTATIONS (only for computer 3666 unless you sync checkpoints )
    #WAS CREATED VIA:
    #    make_hardlink(
    #        "/home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans_BetterAug_WithDropout_50kSamp_T2V_10000000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-9500/transformer",
    #        "/home/jupyter/CleanCode/Github/DiffusionAsShader/diffusion_shader_model_3666Start/transformer",
    #        recursive=True,
    #    )
    MODEL_PATH="./diffusion_shader_model_3666Start"

    MODEL_PATH="./diffusion_shader_model_start"



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
          --dataloader_num_workers 2 \
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
