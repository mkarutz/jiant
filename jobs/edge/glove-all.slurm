#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-0
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH --gres=gpu:v100:1
#SBATCH -A punim0478
#SBATCH -o ../log/edge/ontonotes/glove-all.log
mkdir -p   ../log/edge/ontonotes/

source ~/.bashrc
source ../modules.sh
source user_config.sh
conda activate jiant
python main.py --config_file config/edgeprobe/ontonotes/glove-all.conf
