#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR=[Path to Image]
export MODEL_DIR=[Path to 7B Base Model]
export JSON_PATH=[Path to training json]
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=1
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

LEARNING_RATE=5e-5
VISION_TOWER=openai/clip-vit-large-patch14-336

python -m torch.distributed.launch \
    --master_port=1234 \
    --nproc_per_node=$GPUS_PER_NODE \
    finetune_lora_dpo.py \
    --do_train \
    --base_model_name $MODEL_DIR \
    --per_device_train_batch_size 2 \
    --image_folder $DATA_DIR \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --data_path $JSON_PATH \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --freeze_mm_mlp_adapter True \
    --model_max_length 4096 \
    --gradient_accumulation_steps 1 \
    --logging_steps 50 \
    --output_dir="checkpoints/amp_13B" \
    --warmup_steps 20 \
    --report_to tensorboard \
    --ddp_backend "nccl" \
    --ddp_find_unused_parameters False \
    --resume_from_training False \
    --bits 4 \
    --bf16 True\
    --logging_first_step \
    --max_steps 3000 \
    --num_levels 4