#!/bin/bash
home_dir=./transformed_datasets/german-noise-0.1
data_dir=${home_dir}/Train


output_dir=./transformed_datasets/german-noise-0.1/lambdarank

mkdir $data_dir
mkdir $output_dir

python lambdarank.py --train=${home_dir}/full/train.pkl --test=${home_dir}/full/test.pkl --valid=${home_dir}/full/valid.pkl --comment=full --output_dir=$output_dir

sizes=("50" "250" "500" "2k" "5k" "25k" "50k" "250k")

for size in "${sizes[@]}";
do
  python lambdarank.py --train=${data_dir}/partial_train_"${size}".pkl --test=${home_dir}/full/test.pkl --valid=${data_dir}/partial_valid_"${size}".pkl --comment="${size}" --output_dir=$output_dir
done

home_dir=./transformed_datasets/mslr-noise-0.1
data_dir=${home_dir}/Train


output_dir=./transformed_datasets/mslr-noise-0.1/lambdarank

mkdir $data_dir
mkdir $output_dir

python lambdarank.py --train=${home_dir}/full/train.pkl --test=${home_dir}/full/test.pkl --valid=${home_dir}/full/valid.pkl --comment=full --output_dir=$output_dir

sizes=("1k" "4k" "12k" "36k" "120k")

for size in "${sizes[@]}";
do
  python lambdarank.py --train=${data_dir}/partial_train_"${size}".pkl --test=${home_dir}/full/test.pkl --valid=${data_dir}/partial_valid_"${size}".pkl --comment="${size}" --output_dir=$output_dir
done