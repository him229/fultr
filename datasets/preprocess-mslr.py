import numpy as np
from distutils import file_util, dir_util
import shutil
from tqdm import tqdm
import os
import random
import math
from sklearn.preprocessing import RobustScaler
from utils import serialize, unserialize
from baseline import read_rank_dataset
import argparse


def build_dataset(path, feature_num=None):
    features, rels = {}, {}
    for label, qid, feature, cost in tqdm(read_rank_dataset(path)):
        if qid not in features:
            features[qid] = []
            rels[qid] = []
        if feature_num is None:
            length = len(features)
        else:
            length = feature_num
        feature = [feature.get(i + 1, 0.0) for i in range(length)]
        features[qid].append(feature)
        rels[qid].append(label)
    qids = sorted(features.keys())
    features = [np.array(features[i], dtype=np.float32) for i in qids]
    rels = [np.array(rels[i], dtype=np.float32) for i in qids]
    return (features, rels)


def transform_binary(dataset, threshold=2.5):
    feats, rels = dataset
    rels = [(rel > threshold).astype(np.float32) for rel in rels]
    return (feats, rels)


def filter_relevance(dataset):
    feats, rels = dataset
    filtered = [i for i in range(len(feats)) if (rels[i] > 0.0).any()]
    feats = [feats[i] for i in filtered]
    rels = [rels[i] for i in filtered]
    return feats, rels


def take_absolute(dataset, col):
    feats, rels = dataset
    for i in range(len(feats)):
        feats[i][:, col] = np.absolute(feats[i][:, col])
    return feats, rels


def transform_log_features(dataset, log_features):
    feats, rels = dataset
    for i in range(len(feats)):
        for feature_id in log_features:
            feats[i][:, feature_id] = np.log1p(feats[i][:, feature_id])
    return feats, rels


def build_normalizer(dataset):
    feats, rels = dataset
    all_features = np.concatenate(feats)
    normalizer = RobustScaler(with_centering=False, copy=False, quantile_range=(5.0, 95.0))
    normalizer.fit(all_features)
    return normalizer


def normalize(dataset, normalizer):
    feats, rels = dataset
    feats = [normalizer.transform(feat) for feat in feats]
    return feats, rels


def filter_document(dataset):
    feats, rels = dataset
    for i in range(len(feats)):
        feat, rel = feats[i], rels[i]
        if len(rel) < 20:
            continue
        else:
            ids = np.arange(len(rel))
            rel_docs, nonrel_docs = ids[rel != 0].tolist(), ids[rel == 0].tolist()
            if len(rel_docs) > 3:
                rel_docs = random.sample(rel_docs, 3)
            if len(nonrel_docs) > 20 - len(rel_docs):
                nonrel_docs = random.sample(nonrel_docs, 20 - len(rel_docs))
            ids = rel_docs + nonrel_docs
            feats[i] = np.array([feat[i] for i in ids])
            rels[i] = np.array([rel[i] for i in ids])
    return feats, rels


def filter_candidate(dataset, threshold=20):
    feats, rels = dataset
    feats = [feat for feat in feats if len(feat) >= 20]
    rels = [rel for rel in rels if len(rel) >= 20]
    return feats, rels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_directory", default=None)
    parser.add_argument("--output_directory", default=None)
    parser.add_argument("--no_log_features", action='store_true')
    args = parser.parse_args()
    if args.output_directory is None:
        root_directory = "../transformed_datasets/mslr"
        noise_root_directory = "../transformed_datasets/mslr-noise-0.1"
    else:
        root_directory = os.path.join(args.output_directory, "normal")
        noise_root_directory = os.path.join(args.output_directory, "noise")
    raw_directory = os.path.join(root_directory, "raw")
    train_raw_path = os.path.join(raw_directory, "train.txt")
    valid_raw_path = os.path.join(raw_directory, "vali.txt")
    test_raw_path = os.path.join(raw_directory, "test.txt")

    train_data = build_dataset(train_raw_path)
    train_data = transform_binary(train_data)
    train_data = filter_relevance(train_data)
    print(len(train_data[0]))

    valid_data = build_dataset(valid_raw_path)
    valid_data = transform_binary(valid_data)
    valid_data = filter_relevance(valid_data)
    print(len(valid_data[0]))

    test_data = build_dataset(test_raw_path)
    test_data = transform_binary(test_data)
    test_data = filter_relevance(test_data)
    print(len(test_data[0]))
    if args.no_log_features:
        log_features = []
    else:
        log_features = [10, 11, 12, 13, 14, 40, 41, 42, 43, 44, 90, 91, 92, 93, 94, 127, 133, 134, 135]

    train_data = take_absolute(train_data, 127)
    train_data = transform_log_features(train_data, log_features)
    normalizer = build_normalizer(train_data)
    train_data = normalize(train_data, normalizer)

    valid_data = take_absolute(valid_data, 127)
    valid_data = transform_log_features(valid_data, log_features)
    valid_data = normalize(valid_data, normalizer)

    test_data = take_absolute(test_data, 127)
    test_data = transform_log_features(test_data, log_features)
    test_data = normalize(test_data, normalizer)

    train_data = filter_document(train_data)
    valid_data = filter_document(valid_data)
    test_data = filter_document(test_data)

    train_data = filter_candidate(train_data)
    valid_data = filter_candidate(valid_data)
    test_data = filter_candidate(test_data)

    full_directory = os.path.join(root_directory, "full")
    if not os.path.exists(full_directory):
        os.mkdir(full_directory)
    serialize(train_data, os.path.join(full_directory, "train.pkl"))
    serialize(valid_data, os.path.join(full_directory, "valid.pkl"))
    serialize(test_data, os.path.join(full_directory, "test.pkl"))

    full_directory = os.path.join(noise_root_directory, "full")
    if not os.path.exists(full_directory):
        os.mkdir(full_directory)
    serialize(train_data, os.path.join(full_directory, "train.pkl"))
    serialize(valid_data, os.path.join(full_directory, "valid.pkl"))
    serialize(test_data, os.path.join(full_directory, "test.pkl"))
