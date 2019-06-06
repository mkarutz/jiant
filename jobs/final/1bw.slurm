#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-0
#SBATCH --mem=128G
#SBATCH --partition=deeplearn
#SBATCH --gres=gpu:v100:1
#SBATCH -A punim0478
#SBATCH -o log/final/count2vec.1bw/%a.log
#SBATCH --array=0-19

source ~/.bashrc
source ../modules.sh
source user_config.sh
conda deactivate
conda deactivate
conda activate jiant

i=0

for count2vec_dim in 150; do
for word_embs in "none" "glove"; do
for probing_task in "spr2" "spr1" "dpr" "rel-semeval" "dep-labelling-ewt" "coref-ontonotes" "ner-ontonotes" "nonterminal-ontonotes" "pos-ontonotes" "srl-ontonotes"; do

read -r -d '' overrides <<- EOF
    target_tasks = edges-${probing_task},
    exp_name = final.count2vec.1bw.${count2vec_dim}.${word_embs}.${probing_task},
    count2vec_dim = ${count2vec_dim},
    word_embs = ${word_embs},
    count2vec_collection_dir_path = /home/mkarutz/cstlm-embed/collections/1bw
EOF

if [ $i -eq ${SLURM_ARRAY_TASK_ID} ]; then
    python main.py \
        --config_file config/edgeprobe/final/count2vec.conf \
        --overrides "${overrides}"
fi

i=$((i + 1))

done
done
done
