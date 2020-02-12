data_dir=../transformed_datasets/german-noise-0.1
temp_dir=../transformed_datasets/german-noise-0.1/svmrank
mkdir ${temp_dir}

python svm-rank.py --data_directory ${data_dir} --train_folder Train --normalize --temp_dir ${temp_dir}

data_dir=../transformed_datasets/mslr-noise-0.1
temp_dir=../transformed_datasets/mslr-noise-0.1/svmrank
mkdir ${temp_dir}

python svm-rank.py --data_directory ${data_dir} --train_folder Train --normalize --temp_dir ${temp_dir}