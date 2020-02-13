import numpy as np
from utils import unserialize
from evaluate import *
from baseline import save_svmprop_train, save_svmprop_test, DatasetNormalizer, RankingDataset
import os, shutil, time
from plot import get_stat
from distutils import file_util
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_directory", type=str)
parser.add_argument("--train_folder", type=str, default="Train")
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--temp_dir", type=str)
args = parser.parse_args()

data_directory = args.data_directory
train_dir = os.path.join(data_directory, args.train_folder)
test_dir = os.path.join(data_directory, args.test_folder)
production_dir = os.path.join(data_directory, "production")
svmprop_directory = "../svm_proprank"
learn_path = os.path.join(svmprop_directory, "svm_proprank_learn")
classify_path = os.path.join(svmprop_directory, "svm_proprank_classify")

svmrank_directory = "../svm_rank"
full_learn_path = os.path.join(svmrank_directory, "svm_rank_learn")
full_classify_path = os.path.join(svmrank_directory, "svm_rank_classify")

# loading the full trainset
train_data = unserialize(os.path.join(data_directory, "full", "train.pkl"))
valid_data = unserialize(os.path.join(data_directory, "full", "valid.pkl"))
test_data = unserialize(os.path.join(data_directory, "full", "test.pkl"))
if args.normalize:
    normalizer = DatasetNormalizer(centering=False)
    normalizer.fit(train_data)
else:
    normalizer = None

# saving the full trainset
train_path = os.path.join(train_dir, "train-full.dat")
old_train_path = os.path.join(production_dir, 'train-full.dat')
os.system("ln -s {} {}".format(old_train_path, train_path))
valid_path = os.path.join(train_dir, "valid-full.dat")
old_valid_path = os.path.join(production_dir, 'valid.dat')
os.system("ln -s {} {}".format(old_valid_path, valid_path))
test_path = os.path.join(train_dir, "test-full.dat")
old_test_path = os.path.join(production_dir, 'test.dat')
os.system("ln -s {} {}".format(old_test_path, test_path))

train_range = ["50", "250", "500", "2k", "5k"]
search_range = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
group_id = 14

# train_range = ["359", "1k", "4k", "12k", "36k", "120k"]
# search_range = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
# group_id = 50


def train_grid_search(search_range, metric, train_dir, prefix, mode='max', full=True):
    train_path = os.path.join(train_dir, "train-{}.dat".format(prefix))
    valid_path = os.path.join(train_dir, "valid-{}.dat".format(prefix))
    assert os.path.exists(train_path)
    assert os.path.exists(valid_path)
    if mode == 'min':
        sign = -1
    elif mode == 'max':
        sign = 1
    else:
        raise NotImplementedError
    best, best_metric = None, -1e10
    for c in search_range:
        model_path = os.path.join(data_directory, args.train_folder, "model-{}-{}.dat".format(prefix, c))
        prediction_path = os.path.join(data_directory, args.train_folder, "predictions-valid-{}-{}".format(prefix, c))
        c_str = str(c)
        if not os.path.exists(model_path):
            if full:
                os.system("{} -c {} {} {} >> /dev/null".format(full_learn_path, c_str, train_path, model_path))
            else:
                os.system("{} -c {} {} {} >> /dev/null".format(learn_path, c_str, train_path, model_path))
        if not os.path.exists(prediction_path):
            if full:
                os.system("{} {} {} {} >> /dev/null".format(full_classify_path, valid_path, model_path, prediction_path))
            else:
                os.system("{} {} {} {} >> /dev/null".format(classify_path, valid_path, model_path, prediction_path))
        evaluater = PartialEvaluater(valid_path, prediction_path, group_id=group_id)
        stat = get_stat(evaluater)
        print(c, stat[metric])
        if best is None or stat[metric] * sign > best_metric * sign:
            best_metric = stat[metric]
            best = c
    print("Best c found is {}, valid performance {}".format(best, best_metric))
    file_util.copy_file(os.path.join(data_directory, args.train_folder, "model-{}-{}.dat".format(
        prefix, best)), os.path.join(data_directory, args.train_folder, "model-{}.dat".format(prefix)))


# train the full model with grid search
if not os.path.exists(os.path.join(data_directory, args.train_folder, "model-full.dat")):
    train_grid_search(search_range, "DCG", train_dir, "full", mode='max', full=True)
else:
    print("Full already exists")

# Train with different dataset sizes
tmp_train_dir = args.temp_dir
print(tmp_train_dir)
if not os.path.exists(tmp_train_dir):
    os.mkdir(tmp_train_dir)

to_validate = True
for train_size in train_range:
    print(train_size)
    train_data = unserialize(os.path.join(data_directory, args.train_folder, "partial_train_{}.pkl".format(train_size)))
    train_dataset = RankingDataset(train_data, normalizer=normalizer)
    if to_validate:
        valid_data = unserialize(
            os.path.join(data_directory, args.train_folder, "partial_valid_{}.pkl".format(train_size)))
        valid_dataset = RankingDataset(valid_data, normalizer=normalizer)
    for to_weight in [True, False]:
        weight_str = "weighted" if to_weight else "unweighted"
        prefix = "{}-{}".format(weight_str, train_size)
        if os.path.exists(os.path.join(data_directory, args.train_folder, "model-{}.dat".format(prefix))):
            print("Already complete")
            continue
        train_path = os.path.join(tmp_train_dir, "train-{}-{}.dat".format(weight_str, train_size))
        save_svmprop_train(train_dataset, train_path, weighted=to_weight)
        if to_validate:
            valid_path = os.path.join(tmp_train_dir, "valid-{}-{}.dat".format(weight_str, train_size))
            save_svmprop_train(valid_dataset, valid_path, weighted=to_weight)
        train_grid_search(search_range, "DCG", tmp_train_dir, prefix, mode='max', full=False)

shutil.rmtree(tmp_train_dir)

test_path = os.path.join(data_directory, args.train_folder, "test-full.dat")
for to_weight in [True, False]:
    weight_str = "weighted" if to_weight else "unweighted"
    for train_size in train_range:
        model_path = os.path.join(data_directory, args.train_folder, "model-{}-{}.dat".format(weight_str, train_size))
        prediction_path = os.path.join(data_directory, args.train_folder,
                                       "predictions-test-{}-{}".format(weight_str, train_size))
        os.system("{} {} {} {}".format(full_classify_path, test_path, model_path, prediction_path))

model_path = os.path.join(data_directory, args.train_folder, "model-full.dat")
test_path = os.path.join(data_directory, args.train_folder, "test-full.dat")
prediction_path = os.path.join(data_directory, args.train_folder, "predictions-test-full")
os.system("{} {} {} {} >> /dev/null".format(full_classify_path, test_path, model_path, prediction_path))
