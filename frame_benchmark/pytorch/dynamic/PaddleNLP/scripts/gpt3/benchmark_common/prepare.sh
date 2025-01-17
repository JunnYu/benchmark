#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"
# update megatron/__init__.py
rm -rf ./megatron/__init__.py
cp __init__.py ./megatron

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/
run_env=$ROOT_DIR/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`

################################# 配置python, 如:
rm -rf $run_env
mkdir $run_env
echo `which python3.7`
ln -s $(which python3.7)m-config  $run_env/python3-config
#ln -s /usr/local/python3.7.0/lib/python3.7m-config /usr/local/bin/python3-config
ln -s $(which python3.7) $run_env/python
ln -s $(which python3.7) $run_env/python3
ln -s $(which pip) $run_env/pip
ln -s $(which pip) $run_env/pip3

export PATH=$run_env:${PATH}

#pip install -U pip
echo `pip --version`
echo `python3-config --help`

# echo `pip --version`
# python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
#     --remote-path frame_benchmark/pytorch_req/pytorch_191/ \
#     --local-path ./  \
#     --mode download
# ls
unset https_proxy && unset http_proxy
# pip install -U pip
wget https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install torch-1.9.1+cu111-cp37-cp37m-linux_x86_64.whl
pip install regex pybind11 Ninja -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "https_proxy $HTTPS_PRO" 
echo "http_proxy $HTTP_PRO" 
export https_proxy=$HTTP_PRO
export http_proxy=$HTTP_PRO
echo $http_proxy  $https_proxy 
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com
echo "---------clone apex:"
git clone https://github.com/NVIDIA/apex 
ls
cd ./apex
# 20221109,apex升级后会报错
git checkout -b 1d7711100bb58dc761a2fad89f30d41239450f58 1d7711100bb58dc761a2fad89f30d41239450f58 
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd -
pip list
unset http_proxy && unset https_proxy 
echo "*******prepare benchmark end***********"


# 下载训练数据

if [ -d data ]
then
  rm -rf data
fi

wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/dataset/gpt-benchmarkdata.tar.gz
tar -zxvf gpt-benchmarkdata.tar.gz


if [ -d token_files ]
then
  rm -rf token_files
fi

mkdir token_files && cd token_files
wget http://paddlenlp.bj.bcebos.com/models/transformers/gpt/gpt-en-vocab.json
wget http://paddlenlp.bj.bcebos.com/models/transformers/gpt/gpt-en-merges.txt
cd -

echo "*******prepare benchmark end***********"
