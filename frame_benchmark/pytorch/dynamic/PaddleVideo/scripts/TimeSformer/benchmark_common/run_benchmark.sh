#!/usr/bin/env bash
# Test training benchmark for a model.
# 在TimeSformer目录下执行
# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_name} ${run_mode} ${fp_item} ${bs_item} ${max_iter} ${num_workers}
function _set_params(){
    model_item=${1:-"TimeSformer"}   # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    base_batch_size=${2:-"1"}       # (必选) 每张卡上的batch_size
    fp_item=${3:-"fp32"}            # (必选) fp32|fp16
    run_process_type=${4:-"MultiP"} # (必选) 单进程 SingleP|多进程 MultiP
    run_mode=${5:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${6:-"N1C8"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C8 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="TimeSformer"          # (必选) 模型套件的名字
    speed_unit="instance/sec"         # (必选)速度指标单位
    skip_steps=10                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key=""             # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_epoch=${7:-"1"}                # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件  或是max_epoch
    num_workers=${8:-"4"}             # (可选)
    #   以下为通用拼接log路径，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 切格式不要改动,与平台页面展示对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    # TimeSformer_TimeSformer_bs*_fp*_*P_*_N*C*_log
    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed
    if [ ${profiling} = "true" ];then
            add_options="profiler_options=/"batch_range=[50, 60]; profile_path=model.profile/""
            log_file=${profiling_log_file}
        else
            add_options=""
            log_file=${train_log_file}
    fi
}
function _analysis_log(){
    python analysis_log.py ${model_item} ${log_file} ${speed_log_file} ${device_num}
}
function _train(){
    batch_size=${base_batch_size}  # 如果模型跑多卡单进程时,请在_train函数中计算出多卡需要的bs

    PORT=${PORT:-29500}

    echo "current ${model_name} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=${device_num}, batch_size=${batch_size}"
    accu_steps=8  # TimeSformer梯度累加8次
    train_config="--cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    batch_size_all_gpu=`expr ${batch_size} \* ${num_gpu_devices}`  # 所有卡的单次bs之和
    batch_size_all_gpu_acc=`expr ${batch_size_all_gpu} \* ${accu_steps}`  # 所有卡的accu次bs之和
    echo "batch_size_all_gpu = $batch_size_all_gpu"
    echo "batch_size_all_gpu_acc = $batch_size_all_gpu_acc"
    # TRAIN.BATCH_SIZE是全部卡的bs之和
    train_options="TEST.ENABLE False \
                   TRAIN.AUTO_RESUME False \
                   DATA.PATH_TO_DATA_DIR ./annotations \
                   DATA.PATH_PREFIX ./videos \
                   NUM_GPUS ${num_gpu_devices} \
                   TRAIN.BATCH_SIZE ${batch_size_all_gpu} \
                   GLOBAL_BATCH_SIZE ${batch_size_all_gpu_acc} \
                   TIMESFORMER.PRETRAINED_MODEL ./jx_vit_base_p16_224-80ecf9dd.pth \
                   SOLVER.MAX_EPOCH ${max_epoch}"

    nodes="${device_num:1:1}"

    if [[ nodes -gt 1 ]];then
        init_method="env://"
        export MASTER_ADDR="$POD_0_IP"
        export MASTER_PORT="$PORT"
        export NNODES="$nodes"
        export NODE_RANK="$PADDLE_TRAINER_ID"

        echo "MASTER_ADDR="$POD_0_IP""
        echo "MASTER_PORT="$PORT""
        echo "NNODES="$nodes""
        echo "NODE_RANK="$PADDLE_TRAINER_ID""
        train_cmd="python tools/run_net.py --shard_id $NODE_RANK --num_shards $NNODES --init_method $init_method ${train_config} ${train_options}"
    else
        case ${run_process_type} in
        SingleP) train_cmd="python tools/run_net.py ${train_config} ${train_options}" ;;
        MultiP)
            train_cmd="python tools/run_net.py ${train_config} ${train_options}" ;;
        *) echo "choose run_mode(SingleP or MultiP)"; exit 1;
        esac
    fi

#   以下为通用执行命令，无特殊可不用修改
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    if [ ${run_process_type} = "MultiP" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
}
_set_params $@
export model_branch=`git symbolic-ref HEAD 2>/dev/null | cut -d"/" -f 3`
export model_commit=$(git log|head -n1|awk '{print $2}')
export frame_version=`python -c "import torch;print(torch.__version__)"`
echo "---------frame_version is torch ${frame_version}"
echo "---------model_branch is ${model_branch}"
echo "---------model_commit is ${model_commit}"
job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log
