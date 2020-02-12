#!/bin/bash
#SBATCH -J svmrank                         # Job name
#SBATCH -o tmp/svmrank_%j.out                  # Name of stdout output log file (%j expands to jobID)
#SBATCH -e tmp/svmrank_%j.out                  # Name of stderr output log file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 6                                 # Total number of cores requested
#SBATCH --mem=4GB                          # Total amount of (real) memory requested (per node)
#SBATCH -t 12:00:00                          # Time limit (hh:mm:ss)
#SBATCH --partition=default_gpu              # Request partition for resource allocation

mkdir /scratch/datasets
data_dir=/scratch/datasets/zd224-svm
mkdir $data_dir

python svm-rank.py --data_directory /share/thorsten/zd224/german-noise-0.1 --train_folder Train-4 --test_folder Test-4 --normalize --temp_dir ${data_dir}

rm -rf ${data_dir}