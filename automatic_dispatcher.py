import sys
import os
import numpy as np

dataset = "mslr"
sbatch_output_file_name = "{}_n_en1".format(dataset[0])
num_sruns = 12

sbatch_template = '''#!/bin/bash
#SBATCH -J {}                          # Job name
#SBATCH -o saved_logs/{}.out       # Name of stdout output log file (%j expands to jobID)
#SBATCH -e saved_logs/{}.err       # Name of stderr output log file (%j expands to jobID)
#SBATCH -n 3                                 # Total number of cores requested
#SBATCH --mem=4000                           # Total amount of (real) memory requested (per node)
#SBATCH -t 48:00:00                          # Time limit (hh:mm:ss)
#SBATCH --partition=default_gpu              # Request partition for resource allocation


'''.format(sbatch_output_file_name, sbatch_output_file_name, sbatch_output_file_name)


tasks = None
# tasks = [('full', 'baseline', 25.0, 0.0, 1.0)] + [('full', 'ours', 0.5, 1e-05, 1.0), ('full', 'ours', 15.0, 1e-05, 1.0)]

if tasks is None:
    weight_decays = [0.0, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    entropy_regularizers = [1.0]
    lambda_list = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.125]
    lambda_list += [0.5, 1.0, 10.0, 15.0, 25.0, 50.0, 100.0]
    # lambda_list = [250, 500, 750, 1e3, 5e3, 1e4]
    baselines = ["ours"]
    # baselines = ["baseline", "ours"]
    # info = ["full", "partial"]
    info = ["partial"]

    tasks = []
    for t in info:
        for b in baselines:
            for l in lambda_list:
                for w in weight_decays:
                    for e in entropy_regularizers:
                        tasks.append((t, b, l, w, e))

# tasks = [("full", "ours", 0.01, 1e-3, 1.0)]

print("Total number of tasks = ", len(tasks))

num_sruns = min(num_sruns, len(tasks))

# Deciding how many tasks per srun
counts = [0 for i in range(num_sruns)]
curr = 0
for task_num in range(len(tasks)):
    counts[curr] += 1
    curr += 1
    curr = curr % num_sruns

# Deciding how many tasks per GPU
first, last = 0, counts[0]
for srun in range(num_sruns):
    tasks_to_run = list(range(first, last))
    if srun < num_sruns - 1:
        first = last
        last = first + counts[srun+1]

    print("For ", str(srun) + 'th', "sbatch, task list:")
    print(tasks_to_run)

    args_to_pass = str(np.array(tasks)[tasks_to_run].tolist()).replace(
        ' ', '').replace("'", "\\'")
    print(args_to_pass)
    # continue

    command = sbatch_template + 'python run_tasks.py ' + \
        args_to_pass + " " + dataset

    f = open('hoe_script.sh', 'w')
    f.write(command)
    f.close()

    os.system('sbatch --requeue hoe_script.sh')
