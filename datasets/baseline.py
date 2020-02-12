import numpy as np
import torch.utils.data
from sklearn.preprocessing import RobustScaler


def count_single_group(X, y, group_id):
    count = 0
    for qid in range(len(X)):
        rel_ids = y[qid].nonzero()[0]
        groups = np.unique(X[qid][rel_ids, group_id])
        if len(groups) > 1:
            count += 1
    return count / len(X)


def read_rank_dataset(path):
    with open(path) as file:
        for line in file:
            label, line = line.strip().split(' ', maxsplit=1)
            label = float(label)
            line = dict(list(map(lambda x: x.split(':'), line.split())))
            qid = int(line['qid'])
            features = {int(idx): float(value) for idx, value in line.items() if idx.isdigit()}
            cost = 1.0
            if 'cost' in line:
                cost = float(line['cost'])
            yield label, qid, features, cost


class DatasetNormalizer:
    def __init__(self, centering=False):
        self.scaler = RobustScaler(with_centering=centering)

    def fit(self, train_data):
        x, y = train_data
        x = np.stack(x, axis=0)
        train_size, train_docs, train_features = x.shape
        x = x.reshape(train_size * train_docs, train_features)
        self.scaler.fit(x)
        x = self.scaler.transform(x).reshape(train_size, train_docs, train_features)
        train_data = (x, y)
        return train_data

    def transform(self, x, y=None):
        if y is None:
            return self.scaler.transform(x)
        else:
            return self.scaler.transform(x), y

    def transform_data(self, data):
        test_x, test_y = data
        test_x = np.stack(test_x, axis=0)
        test_size, test_docs, test_features = test_x.shape
        assert len(self.scaler.scale_) == test_features
        test_x = test_x.reshape(test_size * test_docs, test_features)
        test_x = self.scaler.transform(test_x).reshape(test_size, test_docs, test_features)
        test_data = (test_x, test_y)
        return test_data


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, divide_query=True, normalizer: DatasetNormalizer = None, ranking=False):
        xs, ys = [], []
        for x, y in zip(*dataset):
            if divide_query:
                rel_docs = y.nonzero()[0]
                for rel_doc in rel_docs:
                    if normalizer is not None:
                        new_x = normalizer.transform(x)
                    else:
                        new_x = x
                    new_y = np.zeros(len(y), dtype=np.float)
                    new_y[rel_doc] = y[rel_doc]
                    if ranking:
                        sort_idx = np.argsort(-new_y)
                        new_x = new_x[sort_idx]
                        new_y = new_y[sort_idx]
                    xs.append(new_x.astype(np.float32))
                    ys.append(new_y.astype(np.float32))
            else:
                if normalizer is not None:
                    new_x = normalizer.transform(x)
                else:
                    new_x = x
                new_y = y
                if ranking:
                    sort_idx = np.argsort(-new_y)
                    new_x = new_x[sort_idx]
                    new_y = new_y[sort_idx]
                xs.append(new_x.astype(np.float32))
                ys.append(new_y.astype(np.float32))
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        return self.xs[item], self.ys[item]


def save_svmprop_train(dataset: RankingDataset, path, weighted=True, write_cost=True, only_feature=None):
    with open(path, "w") as output:
        for qid in range(len(dataset)):
            x, y = dataset[qid]
            for doc_id in range(len(x)):
                line = []
                if y[doc_id] > 0:
                    line.append("1")
                else:
                    line.append("0")
                line.append("qid:{}".format(qid))
                if y[doc_id] > 0:
                    if write_cost:
                        if weighted:
                            line.append("cost:{}".format(y[doc_id]))
                        else:
                            line.append("cost:{}".format(1.0))
                if only_feature is None:
                    for feature in range(len(x[doc_id])):
                        line.append("{}:{:.3g}".format(feature + 1, x[doc_id, feature]))
                else:
                    line.append("{}:{:.3g}".format(only_feature + 1, x[doc_id, only_feature]))
                output.write(" ".join(line) + "\n")


def save_svmprop_test(dataset: RankingDataset, input_path, ground_path="/dev/null", weighted=True):
    with open(input_path, "w") as input_file, open(ground_path, "w") as ground_file:
        for qid in range(len(dataset)):
            x, y = dataset[qid]
            for doc_id in range(len(y)):
                input_line, ground_line = [], []
                if doc_id == 0:
                    input_line.append("1")
                else:
                    input_line.append("0")
                if y[doc_id] != 0:
                    ground_line.append("1")
                else:
                    ground_line.append("0")
                input_line.append("qid:{}".format(qid))
                ground_line.append("qid:{}".format(qid))
                if y[doc_id] != 0:
                    if weighted:
                        weight = y[doc_id]
                    else:
                        weight = 1.0
                    ground_line.append("cost:{}".format(weight))
                if doc_id == 0:
                    input_line.append("cost:{}".format(1.0))
                for feature in range(len(x[doc_id])):
                    input_line.append("{}:{:.3g}".format(feature + 1, x[doc_id, feature]))
                    ground_line.append("{}:{:.3g}".format(feature + 1, x[doc_id, feature]))
                input_file.write(" ".join(input_line) + "\n")
                ground_file.write(" ".join(ground_line) + "\n")
