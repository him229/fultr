import numpy as np
from utils import unserialize, serialize
from evaluate import *
from baseline import save_svmprop_train, DatasetNormalizer, RankingDataset
import os, random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--data_directory", default=None)
args = parser.parse_args()

if args.data_directory is None:
    root_directory = "../transformed_datasets/{}".format(args.dataset)
    noise_root_directory = "../transformed_datasets/{}-noise-0.1".format(args.dataset)
else:
    root_directory = os.path.join(args.data_directory, "normal")
    noise_root_directory = os.path.join(args.data_directory, "noise")
data_directory = os.path.join(root_directory, "full")
production_directory = os.path.join(root_directory, "production")
if not os.path.exists(production_directory):
    os.mkdir(production_directory)
svmrank_directory = "../svm_rank"
learn_path = os.path.join(svmrank_directory, "svm_rank_learn")
classify_path = os.path.join(svmrank_directory, "svm_rank_classify")
generate_path = "./generate_clicks_for_dataset.py"

# full information setting
train_data = unserialize(os.path.join(data_directory, "train.pkl"))
test_data = unserialize(os.path.join(data_directory, "test.pkl"))
valid_data = unserialize(os.path.join(data_directory, "valid.pkl"))

if args.dataset == "german":
    normalizer = DatasetNormalizer(centering=False)
    normalizer.fit(train_data)
    data_range = [0.1, 0.5, 1, 5, 10, 50, 100, 500]
elif args.dataset == "mslr":
    normalizer = None
    data_range = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
else:
    raise NotImplemented

full_train_path = os.path.join(production_directory, "train-full.dat")
train_dataset = RankingDataset(train_data, normalizer=normalizer)
save_svmprop_train(train_dataset, full_train_path, write_cost=False)

valid_path = os.path.join(production_directory, "valid.dat")
valid_dataset = RankingDataset(valid_data, normalizer=normalizer, divide_query=False)
save_svmprop_train(valid_dataset, valid_path, write_cost=False)

test_path = os.path.join(production_directory, "test.dat")
test_dataset = RankingDataset(test_data, normalizer=normalizer)
save_svmprop_train(test_dataset, test_path, write_cost=False)

model_path = os.path.join(production_directory, "model-full.dat")
os.system("{} -c 1.0 {} {} >> /dev/null".format(learn_path, full_train_path, model_path))
test_path = os.path.join(production_directory, "train-full.dat")
prediction_path = os.path.join(production_directory, "predictions-train-full")
os.system("{} {} {} {} >> /dev/null".format(classify_path, test_path, model_path, prediction_path))
evaluater = PartialEvaluater(test_path, prediction_path)
print("{:10}: {:.3f}".format('MRR', evaluater.partial_evaluate(reciprocal_rank)))
print("{:10}: {:.3f}".format("DCG", evaluater.partial_evaluate(dcg_rank)))
full_mrr = evaluater.partial_evaluate(reciprocal_rank)

production_ratio = 1.0
X, y = [], []
idxs = list(range(len(train_data[0])))
random.shuffle(idxs)
idxs = idxs[:int(production_ratio * len(train_data[0]) // 100)]
for idx in idxs:
    X.append(train_data[0][idx])
    y.append(train_data[1][idx])

train_path = os.path.join(production_directory, "train-{}.dat".format(production_ratio))
train_dataset = RankingDataset((X, y), normalizer=normalizer)
save_svmprop_train(train_dataset, train_path, weighted=False)

model_path = os.path.join(production_directory, "model-{}.dat".format(production_ratio))
os.system("{} -c 1000.0 {} {} >> /dev/null".format(learn_path, train_path, model_path))

# train set
prediction_path = os.path.join(production_directory, "predictions-{}".format(production_ratio))
os.system("{} {} {} {} >> /dev/null".format(classify_path, full_train_path, model_path, prediction_path))

evaluater = PartialEvaluater(test_path, prediction_path)
production_mrr = evaluater.partial_evaluate(reciprocal_rank)
print(full_mrr, production_mrr)

train_raw_path = os.path.join(production_directory, "train-full.dat")
test_raw_path = os.path.join(production_directory, "test.dat")
valid_raw_path = os.path.join(production_directory, "valid.dat")

prediction_train_path = os.path.join(production_directory, "predictions-{}-train".format(production_ratio))
os.system("{} {} {} {} >> /dev/null".format(classify_path, train_raw_path, model_path, prediction_train_path))

prediction_test_path = os.path.join(production_directory, "predictions-{}-test".format(production_ratio))
os.system("{} {} {} {} >> /dev/null".format(classify_path, test_raw_path, model_path, prediction_test_path))

prediction_valid_path = os.path.join(production_directory, "predictions-{}-valid".format(production_ratio))
os.system("{} {} {} {} >> /dev/null".format(classify_path, valid_raw_path, model_path, prediction_valid_path))

train_data = unserialize(os.path.join(data_directory, "train.pkl"))
valid_data = unserialize(os.path.join(data_directory, "valid.pkl"))
test_data = unserialize(os.path.join(data_directory, "test.pkl"))


def output_results(prediction_path, data_path, Xs, ys):
    ranked_X, ranked_y = [], []
    evaluater = PartialEvaluater(data_path, prediction_path)
    ranked_results = evaluater.rank()
    for qid, result in ranked_results.items():
        X, y = Xs[qid], ys[qid]
        ranked_X.append(X[result])
        ranked_y.append(y[result])
    return ranked_X, ranked_y


ranked_train = output_results(prediction_train_path, train_raw_path, train_data[0], train_data[1])
ranked_test = output_results(prediction_test_path, test_raw_path, test_data[0], test_data[1])
ranked_valid = output_results(prediction_valid_path, valid_raw_path, valid_data[0], valid_data[1])

ranked_train_path = os.path.join(production_directory, "ranked_train.pkl")
ranked_test_path = os.path.join(production_directory, "ranked_test.pkl")
ranked_valid_path = os.path.join(production_directory, "ranked_valid.pkl")

serialize(ranked_train, ranked_train_path)
serialize(ranked_test, ranked_test_path)
serialize(ranked_valid, ranked_valid_path)

train_dir = os.path.join(root_directory, "Train")
test_dir = os.path.join(root_directory, "Test")
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
os.system("python {} {} {} {} {} {} \"{}\" --skip_empty".format(generate_path, ranked_train_path,
                                                                ranked_valid_path, ranked_test_path,
                                                                train_dir, test_dir, data_range))

train_dir = os.path.join(noise_root_directory, "Train")
test_dir = os.path.join(noise_root_directory, "Test")
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
os.system(
    "python {} {} {} {} {} {} \"{}\" --en 0.1".format(generate_path, ranked_train_path, ranked_valid_path,
                                                      ranked_test_path, train_dir, test_dir, data_range))
