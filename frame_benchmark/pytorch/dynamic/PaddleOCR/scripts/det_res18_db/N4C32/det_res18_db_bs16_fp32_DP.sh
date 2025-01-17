model_item=det_res18_db
#max_token
bs_item=16
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N4C32
max_iter=5
num_workers=1

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;
