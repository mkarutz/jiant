#!/bin/bash
sbatch jobs/edge/count2vec-coref.slurm
sbatch jobs/edge/count2vec-ner.slurm
sbatch jobs/edge/count2vec-nonterminal.slurm
sbatch jobs/edge/count2vec-pos.slurm
sbatch jobs/edge/count2vec-srl.slurm
