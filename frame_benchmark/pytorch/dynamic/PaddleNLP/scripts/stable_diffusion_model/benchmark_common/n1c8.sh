model_item=stable_diffusion_model
bs_item=16
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N1C8
max_iter=500
num_workers=8

sed -i '/set\ -xe/d' run_benchmark.sh
# bash prepare.sh;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;

# export FLAG_XFORMERS="False"
# export FLAG_RECOMPUTE="False"