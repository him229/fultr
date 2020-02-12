 dataset_name=mslr
 thetas=(0.99 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
 for theta in "${thetas[@]}";
 do
     python postprocess.py --test_data data/${dataset_name}/test.pkl --prediction_file data/${dataset_name}/prediction.txt --output_dir output/${dataset_name}_estimate_${theta} --group_feat_id 132 --group_feat_threshold 0.03252032399177551 --theta ${theta}
 done

 for theta in "${thetas[@]}";
 do
     python postprocess.py --test_data data/${dataset_name}/test.pkl --prediction_file data/${dataset_name}/prediction.txt --output_dir output/${dataset_name}_true_${theta} --group_feat_id 132 --group_feat_threshold 0.03252032399177551 --theta ${theta} --true_merit
 done

dataset_name=german
thetas=(0.99 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
for theta in "${thetas[@]}";
do
    python postprocess.py --test_data data/${dataset_name}/test.pkl --prediction_file data/${dataset_name}/prediction.txt --output_dir output/${dataset_name}_estimate_${theta} --group_feat_id 14 --theta ${theta}
done

for theta in "${thetas[@]}";
do
    python postprocess.py --test_data data/${dataset_name}/test.pkl --prediction_file data/${dataset_name}/prediction.txt --output_dir output/${dataset_name}_true_${theta} --group_feat_id 14 --theta ${theta} --true_merit
done