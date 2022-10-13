#!/bin/bash

#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J anatomy
#SBATCH -o out/test_anatomy.%J.out
#SBATCH -e err/test_anatomy.%J.err
#SBATCH --mail-user=fernando.zhapacamacho@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00
#SBATCH --mem=100G
#SBATCH --constraint=[cascadelake]

function readJobArrayParams () {
    SOURCE_OWL=${1}
    TARGET_OWL=${2}
    REFERENCE=${3}
    emb_size=${4}
    epochs=${5}
    batch_k=${6}
    batch_a=${7}
    lr=${8}
    margin=${9}
    topk=${10}
    minth=${11}
    maxth=${12}
}

function getJobArrayParams () {
  local job_params_file="params_max_threshold.txt"

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

# Run the code
python training_modelR.py --source $SOURCE_OWL --target $TARGET_OWL --reference $REFERENCE -size $emb_size --epochs $epochs --batch-k $batch_k --batch-a $batch_a --lr $lr --margin $margin --topk $topk -minth $minth -maxth $maxth --aim predict -root data/test


			    
			       
