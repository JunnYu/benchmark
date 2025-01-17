model_item=bert_base_seqlen128
bs_item=96
fp_item=fp16
run_process_type=MultiP
run_mode=DP
device_num=N1C8
max_iter=20
num_workers=1

sed -i '/set\ -xe/d' run_benchmark.sh
bash PrepareEnv.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;