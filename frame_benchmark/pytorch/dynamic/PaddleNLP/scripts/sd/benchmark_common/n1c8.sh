model_item=stable_diffusion_model
bs_item=${BATCH_SIZE}
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N1C8
max_iter=200
num_workers=8

sed -i '/set\ -xe/d' run_benchmark.sh
# bash prepare.sh;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;

