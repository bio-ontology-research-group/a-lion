#!/bin/sh -l
#SBATCH --job-name=alighn
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --mem=25GB
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8


function readJobArrayParams () {
  _SIZE=${1}
  _BK=${2}
  _BA=${3}
  _A1=${4}
  _LR=${5}
  _L1=${6}
  _M=${7}
  _AM=${8}
  organism=${9}
}

function getJobArrayParams () {
  local job_params_file="testR_job-params.txt"

  if [ -z "${SLURM_ARRAY_TASK_ID}" ] ; then
    echo "ERROR: Require job array.  Use '--array=#-#', or '--array=#,#,#' to specify."
    echo " Array #'s are 1-based, and correspond to line numbers in job params file: ${job_params_file}"
    exit 1
  fi

  if [ ! -f "$job_params_file" ] ; then  
    echo "Missing job parameters file ${job_params_file}"
    exit 1
  fi

  readJobArrayParams $(head ${job_params_file} -n ${SLURM_ARRAY_TASK_ID} | tail -n 1)
}

getJobArrayParams


module load anaconda3
module load tensorflow/1.13.1-cuda10.0-cudnn7.6-py3.6
##pip install tqdm --user


file_test_name=testR_size${_SIZE}_bk${_BK}_ba${_BA}_a1${_A1}_lr${_LR}_L1_${_L1}_m${_M}_AM${_AM}
model_file=hyperparameterTraining/$organism/$file_test_name/model_r_
data_file=hyperparameterTraining/$organism/$file_test_name/data_r_
kg1f=data/only_phenotype_classes/${organism}/onlypheno_${organism}_d_edgelist.txt
kg2f=data/only_phenotype_classes/${organism}/onlypheno_${organism}_g_edgelist.txt
alignment=data/only_phenotype_classes/${organism}/train


mkdir -p hyperparameterTraining/$organism/$file_test_name

fold=1
for fold in {1..10}
do
	python3.6 training_modelR.py ${_SIZE} ${model_file}${fold} ${data_file}${fold} \
                               ${kg1f} ${kg2f} ${alignment}${fold} ${_BK} ${_BA} ${_A1} ${_L1} ${_LR} ${_M} ${_AM}

done


