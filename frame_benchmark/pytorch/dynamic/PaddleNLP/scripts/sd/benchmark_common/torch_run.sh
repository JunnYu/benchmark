# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

git clone https://github.com/JunnYu/benchmark -b add_sd_diffusers_benchmark
cd benchmark/frame_benchmark/pytorch/dynamic/PaddleNLP/scripts/stable_diffusion_model/benchmark_common

# 准备权重和数据，prepare
# sh prepare.sh

export BATCH_SIZE=64
export FLAG_RECOMPUTE=True
sh n1c1.sh

# ps -ef|grep train_txt2img_laion400m_trainer.py|grep -v grep|cut -c 9-15|xargs kill -9