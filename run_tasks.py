import os
import sys

command = '''python run_hyperparams.py'''

for t, b, l, w, e in list(eval(sys.argv[1])):
    if sys.argv[2] == "mslr":
        position_bias_power = 1.0
        last = '''--early_stopping \
        --entreg_decay "0.6" \
        --partial_train_data "GermanCredit/MSLR_Fold1/train-noise-0.1/partial_train_12k.pkl" \
        --partial_val_data "GermanCredit/MSLR_Fold1/train-noise-0.1/partial_valid_12k.pkl" \
        --partial_test_data "GermanCredit/MSLR_Fold1/test/partial_test_12k.pkl" \
        --full_train_data "GermanCredit/MSLR_Fold1/full/train.pkl" \
        --full_val_data "GermanCredit/MSLR_Fold1/full/valid.pkl" \
        --full_test_data "GermanCredit/MSLR_Fold1/full/test.pkl" \
        --group_feat_threshold "0.03252032399177551" \
        --group_feat_id "132" --input_dim "136"'''
        if b == "baseline":
            baseline_folder = "baseline_mslr_gp132_try6"
            final = '''{} [{}] {} \
                --fullinfo {} \
                --weight_decay {} \
                --entropy_regularizer {} \
                --hyperparam_folder {} \
                --experiment_prefix tuning_{} \
                --log_dir "runs/new/try25_{}" \
                --position_bias_power {} \
                --no-weighted \
                --disparity_type ashudeep'''.format(
                    command, l, last,
                    t, w, e,
                    baseline_folder, baseline_folder, baseline_folder,
                    position_bias_power)
        elif b == "ours":
            ours_folder = "mslr_gp132_try6_sq_en1"
            final = '''{} [{}] {} \
                --fullinfo {} \
                --weight_decay {} \
                --entropy_regularizer {} \
                --hyperparam_folder {} \
                --experiment_prefix tuning_{} \
                --log_dir "runs/new/try25_{}" \
                --position_bias_power {} \
                --noise \
                --en 0.1'''.format(
                    command, l, last,
                    t, w, e,
                    ours_folder, ours_folder, ours_folder,
                    position_bias_power)
            print(final)
        else:
            print("PROBLEM WITH BASELINE")
            exit(0)
    elif sys.argv[2] == "german":
        last = '''--early_stopping \
        --entreg_decay "0.6"'''
        if b == "baseline":
            baseline_folder = "baseline_gp14_sp0_try6"
            final = '''{} [{}] {} \
                --fullinfo {} \
                --weight_decay {} \
                --entropy_regularizer {} \
                --hyperparam_folder {} \
                --experiment_prefix tuning_{} \
                --log_dir "runs/new/try25_{}" \
                --no-weighted \
                --disparity_type ashudeep'''.format(
                    command, l, last, 
                    t, w, e,
                    baseline_folder, baseline_folder, baseline_folder)
        elif b == "ours":
            ours_folder = "gp14_sp0_try6_sq_en1"
            final = '''{} [{}] {} \
                --fullinfo {} \
                --weight_decay {} \
                --entropy_regularizer {} \
                --hyperparam_folder {} \
                --experiment_prefix tuning_{} \
                --log_dir "runs/new/try25_{}" \
                --noise \
                --en 0.1'''.format(
                    command, l, last,
                    t, w, e,
                    ours_folder, ours_folder, ours_folder)
        else:
            print("PROBLEM WITH BASELINE")
            exit(0)
        if t == "full":
            final += " --write_losses_interval {} --evaluate_interval {}".format(
                50, 900)
    if final is None:
        print("PROBLEM WITH FINAL COMMAND")
        exit(0)
    os.system(final)
    print("Finished Task")

print("SBATCH# " + str(sys.argv[2]) + " finished assigned tasks.")
