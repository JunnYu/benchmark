#!/usr/bin/env bash
# Test training benchmark for a model.
# Usage: CUDA_VISIBLE_DEVICES=xxx bash run_benchmark.sh ${model_name} ${run_mode} ${fp_item} ${bs_item} ${max_iter} ${num_workers}
function _set_params(){
    model_item=${1:-"model_item"}   # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    base_batch_size=${2:-"2"}       # (必选) 每张卡上的batch_size
    fp_item=${3:-"fp32"}            # (必选) fp32|fp16
    run_process_type=${4:-"MultiP"} # (必选) 单进程 SingleP|多进程 MultiP
    run_mode=${5:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${6:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C8 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="DeepLearningExamples"          # (必选) 模型套件的名字
    speed_unit="step/s"         # (必选)速度指标单位
    skip_steps=10                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="it/s:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key=""             # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_iter=${7:-"100"}                # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件  或是max_epoch
    num_workers=${8:-"3"}             # (可选)
    #   以下为通用拼接log路径，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 切格式不要改动,与平台页面展示对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    # mmsegmentation_fastscnn_bs2_fp32_MultiP_DP_N1C1_log
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
    cmd="python analysis_log.py ${model_item} ${log_file} ${speed_log_file} ${device_num} ${fp_item} ${base_batch_size}"
    echo $cmd
    eval $cmd
}
function _train(){
    batch_size=${base_batch_size}  # 如果模型跑多卡但进程时,请在_train函数中计算出多卡需要的bs
    echo "current ${model_name} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=${device_num}, batch_size=${batch_size}"
    DATA_DIR_PHASE2=/workspace/bert/data/pretrain/phase2/bin_size_64/parquet
    DATA_DIR_PHASE1=/workspace/bert/data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en # change this for other datasets
    DATA_DIR_PHASE=/workspace/bert/data/pretrain/phase1/unbinned/parquet
    RESULTS_DIR=/workspace/bert/results/
    CHECKPOINTS_DIR=${RESULTS_DIR}/checkpoints
    BERT_CONFIG=/workspace/bert/bert_config_base.json

    PREC=""
    if [ ${fp_item} = "fp16" ];then
        PREC="--fp16"
    fi

    ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
    ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
    CHECKPOINT=""
    INIT_CHECKPOINT=""
    gradient_accumulation_steps=$(expr 67584 \/ $base_batch_size \/ $num_gpu_devices)
    train_batch_size=$(expr 67584 \/ $num_gpu_devices)   # total batch_size per gpu
    
    CMD=" --input_dir=$DATA_DIR_PHASE1"
    CMD+=" --output_dir=$CHECKPOINTS_DIR"
    CMD+=" --config_file=${BERT_CONFIG}"
    CMD+=" --train_batch_size=${train_batch_size}"
    CMD+=" --max_seq_length=128"
    CMD+=" --max_predictions_per_seq=20"
    CMD+=" --max_steps=${max_iter}"
    CMD+=" --warmup_proportion=0.2843"
    CMD+=" --num_steps_per_checkpoint=200"
    CMD+=" --learning_rate=6e-3"
    CMD+=" --seed=12439"
    CMD+=" --bert_model=bert-base-uncased"
    CMD+=" $PREC"
    CMD+=" --gradient_accumulation_steps=${gradient_accumulation_steps}"
    CMD+=" $CHECKPOINT"
    CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
    CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
    CMD+=" $INIT_CHECKPOINT"
    CMD+=" --do_train"
    CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json"
    CMD+=" --disable_progress_bar"
    rm -rf /workspace/bert/results/checkpoints/
    case ${run_process_type} in
    SingleP) train_cmd="python3 -m torch.distributed.launch --nproc_per_node=1  /workspace/bert/run_pretraining.py ${CMD}" ;;
    MultiP)
        train_cmd="python3 -m torch.distributed.launch  --nproc_per_node=8 /workspace/bert/run_pretraining.py  ${CMD}" ;;
    *) echo "choose run_mode(SingleP or MultiP)"; exit 1;
    esac
#   以下为通用执行命令，无特殊可不用修改
    echo ${train_cmd}
    timeout 40m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    if [ ${run_process_type} = "MultiP" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
    echo ${train_cmd} >> ${log_file}
    cat  ${log_file}
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
}
_set_params $@
export frame_version=`python -c "import torch;print(torch.__version__)"`
echo "---------frame_version is torch ${frame_version}"
echo "---------model_branch is ${model_branch}"
echo "---------model_commit is ${model_commit}"
job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log
