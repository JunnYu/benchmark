git clone https://github.com/JunnYu/benchmark -b add_sd_diffusers_benchmark
cd benchmark/frame_benchmark/pytorch/dynamic/PaddleNLP/scripts/stable_diffusion_model/benchmark_common

# 准备权重和数据，prepare
# sh prepare.sh

export BATCH_SIZE=10
# 是否使用facebook的xformers
export FLAG_XFORMERS=False
# 是否使用原生的torch2.0的SDP
export FLAG_SDP=True
export FLAG_RECOMPUTE=True
sh n1c1.sh

# ps -ef|grep train_txt2img_laion400m_trainer.py|grep -v grep|cut -c 9-15|xargs kill -9