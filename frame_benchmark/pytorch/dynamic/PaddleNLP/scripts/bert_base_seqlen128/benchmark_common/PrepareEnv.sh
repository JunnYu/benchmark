echo "*******prepare benchmark start ***********"
# /paddle/DeepLearningExamples/PyTorch/LanguageModeling/BERT
ROOT_DIR=$PWD

echo "--------ROOT_DIR:" $ROOT_DIR
cd PyTorch/LanguageModeling/BERT

ln -s  $PWD /workspace/bert

mkdir /workspace/bert/data

export BERT_PREP_WORKING_DIR=/workspace/bert/data

apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract


apt-get install libb64-0d

cd /workspace/bert
pip install --no-cache-dir \
 tqdm boto3 requests six ipdb h5py nltk progressbar onnxruntime \
 git+https://github.com/NVIDIA/dllogger wget

apt-get install -y iputils-ping

#download data, extract data
# bash data/create_datasets_from_start.sh
cd /workspace/bert/data/
# hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en
wget -q https://bj.bcebos.com/paddlenlp/data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.zip
unzip hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5.zip

cd ${ROOT_DIR}

cp bert_config_base.json /workspace/bert

echo "*******prepare benchmark end***********"
