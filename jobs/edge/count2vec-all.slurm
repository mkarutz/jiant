#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-0
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH --gres=gpu:v100:1
#SBATCH -A punim0478
#SBATCH -o log/count2vec_all_%a.log
#SBATCH --array=0-29

source ~/.bashrc
source ../modules.sh
source user_config.sh
conda deactivate
conda deactivate
conda activate jiant

i=0

for count2vec_dim in 150; do
for word_embs in "none" "glove"; do
for probing_task in "spr2" "spr1" "dpr" "rel-semeval" "dep-labelling-ewt" "coref" "ner" "nonterminal" "pos" "srl"; do

read -r -d '' overrides <<- EOF
    count2vec_dim = ${count2vec_dim},
    word_embs = ${word_embs},
    target_tasks = edges-${probing_task}-ontonotes,
    exp_name = count2vec.${count2vec_dim}.${word_embs}.${probing_task}
EOF

if [ $i -eq ${SLURM_ARRAY_TASK_ID} ]; then
    python main.py \
        --config_file config/edgeprobe/ontonotes/count2vec.conf \
        --overrides "${overrides}"
fi

i=$((i + 1))

done
done
done
