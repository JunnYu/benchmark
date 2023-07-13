export BATCH_SIZE=64
export MAX_ITER=200000

export FLAG_RECOMPUTE=1
export FLAG_BENCHMARK=1
export FLAG_USE_EMA=1
export OUTPUT_DIR="bf16_torch"

node_num=${PADDLE_TRAINERS_NUM}
num_workers=8
node_rank=${PADDLE_TRAINER_ID}
master_addr=${POD_0_IP}
master_port=14233

torchrun --nnodes ${node_num} --nproc_per_node ${num_workers} --node_rank ${node_rank} --master_addr ${master_addr} --master_port ${master_port} train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --logging_steps 10 \
    --resolution 256 \
    --save_steps 10000 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 8 \
    --pretrained_model_name_or_path ./CompVis-stable-diffusion-v1-4 \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --disable_tqdm True \
    --overwrite_output_dir True \
    --optim adamw_torch \
    --tf32 True  \
    --image_logging_steps 1000 \
    --bf16 True \
    --ddp_find_unused_parameters False