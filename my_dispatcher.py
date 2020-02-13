import os
import time
import shutil
import random
from argparse import ArgumentParser
from utils import serialize, add_bool_arg

num_sruns = 25
task_per_srun = 1

parser = ArgumentParser()

parser.add_argument("dataset", default="mslr")
parser.add_argument("--root_directory", type=str)
parser.add_argument("--train_folder", type=str, default="Train")
parser.add_argument("--policy_folder", type=str, default="policy")
parser.add_argument('--utility', action='store_true', default=False)
parser.add_argument('--skyline', action='store_true', default=False)
parser.add_argument("--mlp", action='store_true', default=False)
parser.add_argument('--ashudeep', action='store_true', default=False)
parser.add_argument('--unweighted', action='store_true', default=False)
parser.add_argument('--unweighted_fairness', action='store_true', default=False)
parser.add_argument('--masked', action='store_true', default=False)
parser.add_argument("--noise", action='store_true', default=False)
parser.add_argument("--comment", default=None, type=str)
parser.add_argument("--lambdas", default="[0.0,0.005,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0]", type=str)
parser.add_argument("--train_sizes", default="['full','12k']", type=str)
parser.add_argument("--depend", type=int, default=None)
parser.add_argument("--memory_limit", type=int, default=3)
add_bool_arg(parser, "gpu", default=True)
args = parser.parse_args()

lambdas = eval(args.lambdas)
train_sizes = eval(args.train_sizes)
root_directory = args.root_directory

task_name = [args.dataset]
if args.utility:
    task_name.append('utility')
if args.mlp:
    task_name.append('mlp')
if args.ashudeep:
    task_name.append('ashudeep')
if args.unweighted:
    task_name.append('unweighted')
if args.unweighted_fairness:
    task_name.append('unweighted_fairness')
if args.masked:
    task_name.append('masked')
if args.noise:
    task_name.append('noise')
if args.comment is not None:
    task_name.append(args.comment)
task_name = "-".join(task_name)

arguments = {}

if 'german' in args.dataset.lower():
    arguments['group_feat_id'] = 14
    arguments['group_feat_threshold'] = None
    arguments['input_dim'] = 58
    if 'german-noise-0.1' in args.dataset.lower():
        if not args.noise:
            arguments['noise'] = True
            arguments['en'] = 0.1
elif 'mslr' in args.dataset.lower():
    arguments['group_feat_id'] = 132
    arguments['group_feat_threshold'] = 0.03252032399177551
    arguments['input_dim'] = 136
    if 'mslr-1.5' in args.dataset.lower():
        arguments['position_bias_power'] = 1.5
    elif 'mslr-noise-0.1' in args.dataset.lower():
        if not args.noise:
            arguments['noise'] = True
            arguments['en'] = 0.1043956043956044
else:
    raise NotImplementedError

full_train_data = os.path.join(root_directory, "full", "train.pkl")
full_valid_data = os.path.join(root_directory, "full", "valid.pkl")

arguments['fullinfo'] = 'partial'
arguments['full_test_data'] = os.path.join(root_directory, "full", 'test.pkl')
arguments['eval_other_train_location'] = full_train_data
arguments['eval_weighted_val_location'] = full_valid_data
arguments['eval_other_train'] = True
arguments['eval_weighted_val'] = True

if args.ashudeep:
    arguments['weighted'] = False
    arguments['disparity_type'] = 'ashudeep'
elif args.unweighted:
    arguments['weighted'] = False
if args.unweighted_fairness:
    arguments['unweighted_fairness'] = True
if args.masked:
    arguments['mask_group_feat'] = True

if args.mlp:
    arguments['lr'] = 1e-4
    arguments['optimizer'] = 'Adam'
    arguments['hidden_layer'] = 1
    arguments['evaluate_interval'] = 3000
else:
    arguments['lr'] = 1e-3
    arguments['optimizer'] = 'SGD'
    arguments['evaluate_interval'] = 1000

arguments['gpu'] = args.gpu
if args.gpu:
    arguments['batch_size'] = 64
else:
    arguments['batch_size'] = 4
    arguments['evaluate_interval'] *= 3

arguments['early_stopping'] = True
arguments['use_baseline'] = True
arguments['progressbar'] = False
arguments['summary_writing'] = True
arguments['log_dir'] = "runs/{}".format(task_name)

if args.utility:
    arguments['validation_deterministic'] = True
    arguments['evaluation_deterministic'] = True
    arguments['stop_patience'] = 20
    if args.skyline:
        hyper_parameters = {'entropy_regularizer': [1.0], 'entreg_decay': [1.0],
                        'weight_decay': [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0]}
        arguments['evaluate_interval'] //= 3
    else:
        hyper_parameters = {'entropy_regularizer': [1.0], 'entreg_decay': [0.2],
                            'weight_decay': [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0]}
else:
    arguments['validation_deterministic'] = False
    arguments['evaluation_deterministic'] = False
    arguments['stop_patience'] = 20
    hyper_parameters = {'entropy_regularizer': [1.0], 'entreg_decay': [0.2],
                        'weight_decay': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0]}

tasks = []


def generate_tasks(hyperparam_dict: dict, task_list, current_task):
    if len(hyperparam_dict) == 0:
        task_list.append(dict(current_task))
        return
    key, values = hyperparam_dict.popitem()
    for value in values:
        generate_tasks(hyperparam_dict, task_list, current_task + [(key, value)])
    hyperparam_dict[key] = values


policy_dir = os.path.join(root_directory, args.policy_folder)
if not os.path.exists(policy_dir):
    os.makedirs(policy_dir)

for train_size in train_sizes:
    for lamb in lambdas:
        tune_tasks = []
        current_task = list(arguments.items())
        if train_size == 'full':
            partial_train_data = full_train_data
            partial_valid_data = full_valid_data
        else:
            partial_train_data = os.path.join(root_directory, args.train_folder, "partial_train_{}.pkl".format(train_size))
            partial_valid_data = os.path.join(root_directory, args.train_folder, "partial_valid_{}.pkl".format(train_size))
        current_task.append(('partial_train_data', partial_train_data))
        current_task.append(('partial_val_data', partial_valid_data))
        current_task.append(('lambda_list', str([lamb])))
        generate_tasks(hyper_parameters, tune_tasks, current_task)
        for task in tune_tasks:
            hyperparam_folder = os.path.join(policy_dir, "{}_{}_{}".format(task_name, train_size, lamb),
                                             "{}_{}_{}".format(task['weight_decay'],
                                                               task['entropy_regularizer'],
                                                               task['entreg_decay']))
            task['hyperparam_folder'] = hyperparam_folder
            task["experiment_prefix"] = "{}".format(train_size)
            completed = False
            if os.path.exists(hyperparam_folder):
                for entry in os.scandir(hyperparam_folder):
                    if entry.name.startswith('plt_data') and entry.name.endswith('json'):
                        completed = True
                        break
            if not completed:
                tasks.append(task)

random.shuffle(tasks)

stime = time.strftime("%m-%d-%H-%M")
args_directory = "script/{}-{}-args".format(task_name, stime)
if os.path.exists(args_directory):
    shutil.rmtree(args_directory)
os.makedirs(args_directory)

for i, task in enumerate(tasks):
    srun_id = i // task_per_srun
    sub_id = i % task_per_srun
    serialize(task, os.path.join(args_directory, "{}_{}.json".format(srun_id, sub_id)), in_json=True)

sbatch_file = "script/{}-{}.sh".format(task_name, stime)
for i in range(task_per_srun):
    for j in range(len(tasks) // task_per_srun):
        command = "python run_hyperparams.py \"[0.0]\" --args_file {}/{}_{}.json".format(
        args_directory, j, i)
        os.system(command)
