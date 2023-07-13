# unset PADDLE_ELASTIC_JOB_ID
# unset PADDLE_TRAINER_ENDPOINTS
# unset DISTRIBUTED_TRAINER_ENDPOINTS
# unset FLAGS_START_PORT
# unset PADDLE_ELASTIC_TIMEOUT

export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

export FLAG_USE_EMA=0
export FLAG_RECOMPUTE=1
export FLAG_BENCHMARK=1
export FLAG_USE_EMA=1

export OUTPUT_DIR="bf16_torch"
export BATCH_SIZE=64
export MAX_ITER=200000

master_port=14233

nohup torchrun --nnodes ${PADDLE_TRAINERS_NUM} --nproc_per_node 8 --node_rank ${PADDLE_TRAINER_ID} --master_addr ${POD_0_IP} --master_port ${master_port} train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
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
    --tf32 True \
    --bf16 True \
    --ddp_find_unused_parameters False > torch_sd_bf16_2048.log 2>&1 &