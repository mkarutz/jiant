#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-0
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH --gres=gpu:v100:1
#SBATCH -A punim0478
#SBATCH -o log/final/glove/%a.log
#SBATCH --array=0-9

source ~/.bashrc
source ../modules.sh
source user_config.sh
conda deactivate
conda deactivate
conda activate jiant

i=0

for probing_task in "spr2" "spr1" "dpr" "rel-semeval" "dep-labelling-ewt" "coref-ontonotes" "ner-ontonotes" "nonterminal-ontonotes" "pos-ontonotes" "srl-ontonotes"; do



read -r -d '' overrides <<- EOF
    target_tasks = edges-${probing_task},
    exp_name = final.glove.${probing_task}
EOF

if [ $i -eq ${SLURM_ARRAY_TASK_ID} ]; then
    python main.py \
        --config_file config/edgeprobe/final/glove.conf \
        --overrides "${overrides}"
fi

i=$((i + 1))
done
