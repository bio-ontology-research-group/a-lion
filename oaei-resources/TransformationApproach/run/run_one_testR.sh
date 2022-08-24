#!/bin/sh -l
#SBATCH --job-name=alighn
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=8


module load anaconda3
module load tensorflow/1.13.1-cuda10.0-cudnn7.6-py3.6
#pip install tqdm --user

file_test_name=testR_size${1}_bk${2}_ba${3}_a1${4}_lr${5}_L1_${6}_m${7}_AM${8}
organism=$9
model_file=hyperparameterTraining/$organism/$file_test_name/model_r_
data_file=hyperparameterTraining/$organism/$file_test_name/data_r_
kg1f=data/only_phenotype_classes/${organism}/onlypheno_${organism}_d_edgelist.txt
kg2f=data/only_phenotype_classes/${organism}/onlypheno_${organism}_g_edgelist.txt
alignment=data/only_phenotype_classes/${organism}/train


mkdir hyperparameterTraining/$organism/$file_test_name

fold=1
for fold in {1..10}
do
	python3.6 training_modelR.py $1 $model_file$fold $data_file$fold $kg1f $kg2f ${alignment}$fold $2 $3 $4 $6 $5 $7 $8

done

