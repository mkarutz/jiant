#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-0
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH --gres=gpu:v100:1
#SBATCH -A punim0478
#SBATCH -o log/test/count2vec/log.log

source ~/.bashrc
source ../modules.sh
source user_config.sh
conda deactivate
conda deactivate
conda activate jiant

for count2vec_dim in 300; do
for word_embs in "scratch"; do
for probing_task in "rel-semeval"; do

read -r -d '' overrides <<- EOF
    target_tasks = edges-${probing_task},
    exp_name = test.count2vec.${count2vec_dim}.${word_embs}.${probing_task},
    count2vec_dim = ${count2vec_dim},
    word_embs = ${word_embs},
    embeddings_train = 1,
EOF

python main.py \
    --config_file config/edgeprobe/final/count2vec.conf \
    --overrides "${overrides}"

done
done
done
