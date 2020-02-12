mkdir /scratch/datasets
data_dir=/scratch/datasets/zd224-svm
mkdir $data_dir

python svm-rank.py --data_directory /share/thorsten/zd224/german-noise-0.1 --train_folder Train-4 --test_folder Test-4 --normalize --temp_dir ${data_dir}

rm -rf ${data_dir}