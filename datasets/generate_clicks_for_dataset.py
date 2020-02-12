import sys
sys.path.append('..')
from datareader import DataReader, reader_from_pickle
import argparse
import random
import numpy as np
import os

# dr.data = pkl.load(open("GermanCredit/german_train_rank.pkl", "rb"))
# vdr = YahooDataReader(None)
# vdr.data = pkl.load(open("GermanCredit/german_test_rank.pkl","rb"))

eta = 1.0
ep = 1.0
en = 0.0


def get_propensity(idx=None, cand_set_len=10):
    #     returns the propensity for a position or the whole list
    #     polynomial propensity 1/x
    # cand_set_len = len(dr.data[1][0])
    if idx is None:
        return 1 / np.arange(1, cand_set_len + 1) ** eta
    else:
        return 1.0 / (idx + 1) ** eta


def record_click(prop, rel):
    #     flips a coin to get a click
    prop = prop * ep if rel else prop * en
    return float(np.random.binomial(1, prop))


def rel_to_click(rel):
    return np.asarray(
        [record_click(get_propensity(i),
                      r) for i, r in enumerate(rel)])


def prop_weighted_click(clicks):
    return np.asarray([c / get_propensity(i) for i, c in enumerate(clicks)])


def gen_click_data(rel_data, num_samples=10000, num_clicks=None,
                   skip_empty=True):
    # returns (features, relevances, clicks, propensity_weighted_clicks)
    new_data = [[], [], [], []]
    ids = list(range(len(rel_data[0])))
    clicked_count, sample_count = 0, 0
    relevant_clicks, total_clicks = 0, 0
    while True:
        random.shuffle(ids)
        for idx in ids:
            sample_count += 1
            if (num_clicks is None and sample_count > num_samples) or (
                    num_clicks is not None and clicked_count > num_clicks):
                print("Sample count {}, query count {}, click count {}".format(
                    sample_count, len(new_data[0]), clicked_count))
                print("Relevant click ratio: {}".format(
                    relevant_clicks / total_clicks))
                return tuple(new_data)
            click_data = rel_to_click(rel_data[1][idx])
            if np.sum(click_data) != 0:
                clicked_count += 1
                relevant_clicks += np.sum(np.logical_and(click_data >
                                                         0, rel_data[1][idx] > 0))
                total_clicks += np.sum(click_data > 0)
            elif skip_empty:
                continue
            prop_click_data = prop_weighted_click(click_data)
            new_data[0].append(rel_data[0][idx])
            new_data[1].append(rel_data[1][idx])
            new_data[2].append(click_data)
            new_data[3].append(prop_click_data)


def get_sampled_validations_datasets(
        vdr, num_samples_val=10000, save=False, output_dir=""):
    # generate new data
    val_new_click_data = gen_click_data(vdr.data, num_samples_val)

    new_wvdr = DataReader()
    new_wvdr.data = (val_new_click_data[0], val_new_click_data[3])

    if save:
        name_k = "{}k".format(int(num_samples_val / 1000.0)
                              ) if num_samples_val >= 1000.0 else "{}".format(int(num_samples_val))
        new_wvdr.pickelize_data(outpath=os.path.join(
            output_dir, "partial_validations_{}.pkl".format(name_k)))

    return new_wvdr


def get_sampled_train_datasets(
        dr, num_samples=10000.0, save=False, output_dir=""):
    # generate new data
    new_click_data = gen_click_data(dr.data, num_samples)

    new_wdr = DataReader()
    new_wdr.data = (new_click_data[0], new_click_data[3])

    if save:
        name_k = "{}k".format(int(
            num_samples / 1000.0)) if num_samples >= 1000.0 else "{}".format(int(num_samples))
        new_wdr.pickelize_data(outpath=os.path.join(
            output_dir, "partial_train_{}.pkl".format(name_k)))

    return new_wdr


def get_sampled_datasets(dr, output_path=None, **kwargs):
    # generate new data
    new_click_data = gen_click_data(dr.data, **kwargs)

    new_wdr = DataReader()
    new_wdr.data = (new_click_data[0], new_click_data[3])

    if output_path is not None:
        new_wdr.pickelize_data(outpath=output_path)

    return new_wdr


if __name__ == "__main__":
    # CHANGE these variables to the output of the production ranker
    parser = argparse.ArgumentParser()
    parser.add_argument("production_ranker_train_path", type=str)
    parser.add_argument("production_ranker_valid_path", type=str)
    parser.add_argument("production_ranker_test_path", type=str)
    parser.add_argument("train_dir", type=str)
    parser.add_argument("test_dir", type=str)
    parser.add_argument("dataset_expand", type=str)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--ep", type=float, default=1.0)
    parser.add_argument("--en", type=float, default=0.0)
    parser.add_argument("--skip_empty", action='store_true')
    args = parser.parse_args()

    train_dr = reader_from_pickle(args.production_ranker_train_path)
    valid_dr = reader_from_pickle(args.production_ranker_valid_path)
    test_dr = reader_from_pickle(args.production_ranker_test_path)

    eta = args.eta
    ep = args.ep
    en = args.en

    for i in eval(args.dataset_expand):
        train_samples = int(i * len(train_dr.data[0]))
        valid_samples = int(i * len(valid_dr.data[0]))
        test_samples = int(i * len(test_dr.data[0]))
        name_k = "{}k".format(round(
            train_samples / 1000)) if train_samples >= 1000.0 else "{}".format(int(train_samples))
        train_path = os.path.join(
            args.train_dir, "partial_train_{}.pkl".format(name_k))
        get_sampled_datasets(
            train_dr, train_path, num_clicks=train_samples,
            skip_empty=args.skip_empty)
        valid_path = os.path.join(
            args.train_dir, "partial_valid_{}.pkl".format(name_k))
        get_sampled_datasets(
            valid_dr, valid_path, num_clicks=valid_samples,
            skip_empty=args.skip_empty)
        test_path = os.path.join(
            args.test_dir, "partial_test_{}.pkl".format(name_k))
        get_sampled_datasets(
            test_dr, test_path, num_clicks=test_samples,
            skip_empty=args.skip_empty)
