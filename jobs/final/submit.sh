#!/bin/bash
mkdir -p  log/final/count2vec
mkdir -p  log/final/glove
mkdir -p  log/final/elmo
mkdir -p  log/final/scratch

# sbatch jobs/final/count2vec.slurm
sbatch jobs/final/glove.slurm
sbatch jobs/final/elmo.slurm
sbatch jobs/final/scratch.slurm