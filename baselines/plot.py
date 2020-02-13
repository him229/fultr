import numpy as np
import matplotlib.pyplot as plt
from evaluate import *
import os
import matplotlib
from utils import unserialize

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

data_directory = "/home/zd224/data/relevance"
svmrank_directory = "/home/zd224/code/svm_proprank"
learn_path = os.path.join(svmrank_directory, "svm_proprank_learn")
classify_path = os.path.join(svmrank_directory, "svm_proprank_classify")


def get_stat(evaluater, with_group=False):
    stat = {}
    stat["Avg. Rank"] = evaluater.partial_evaluate(average_rank)
    stat["DCG"] = evaluater.partial_evaluate(dcg_rank)
    stat["WTA"] = evaluater.partial_evaluate(wta_rank)
    if with_group:
        stat["disp1"] = evaluater.compute_group_disparity(disparity_type='disp1')
        stat["disp3"] = evaluater.compute_group_disparity(disparity_type='disp3')
    return stat


marker = ['+', 'o', '.', '*', 'v', 's', '>', 'x']


def plot_curve(performance, attribute, size_range, settings, baselines=None, file_path=None):
    my_marker = marker[:]
    if baselines is not None:
        for name, values in baselines.items():
            plt.plot([values[attribute]] * max(map(len, performance.values())),linestyle='dashed', **settings[name])
    for key, values in performance.items():
        plt.plot([value[attribute] for value in values], **settings[key])
    plt.xticks(np.arange(len(size_range)), size_range)
    plt.xlabel('Number of Training Clicks')
    plt.ylabel('Avg. DCG')

def read_performance(file_path):
    result, valid_perf = None, None
    result = unserialize(file_path)
    result = result[0]
    result, valid_perf = result['test'], result['best_perf']
    stat = {'DCG': result['dcg'], 'Avg. Rank': result['avg_rank']}
    if 'other_disparities' in result:
        for key in result['other_disparities']:
            stat[key] = result['other_disparities'][key][0]
        if 'disp3' in result['other_disparities']:
            stat['disp3'] = result['other_disparities']['disp3'][0]
    if not 'disp3' in stat:
        stat['disp3'] = result['avg_group_asym_disparity']
    return valid_perf, stat


def validate_performance(aim_directory):
    best_result, best_performance, best_hyper = None, -1e6, None
    for directory in os.scandir(aim_directory):
        if directory.is_dir():
            result = None
            for entry in os.scandir(directory.path):
                if entry.name.startswith("plt_data") and entry.name.endswith('.json'):
                    valid_perf, result = read_performance(entry.path)
            if result is not None and valid_perf is not None:
                if best_result is None or valid_perf > best_performance:
                    best_performance = valid_perf
                    best_result = result
                    best_hyper = directory.path
    if best_result is not None:
        return best_result
    else:
        return None


if __name__ == "__main__":
    # evaluate the full model on the full testset
    test_path = os.path.join(data_directory, "Train", "test-full.dat")
    prediction_path = os.path.join(data_directory, "Test", "predictions-test-full")
    evaluater = PartialEvaluater(test_path, prediction_path, group_id=group_id)
    full_performance = get_stat(evaluater)
    print("Full performance: {}".format(full_performance))

    train_range = ["1k", "5k", "10k", "50k", "100k", "500k", "1000k"]
    train_performances = {}
    group_id = 37

    test_path = os.path.join(data_directory, "Train", "test-full.dat")
    for to_weight in [True, False]:
        weight_str = "weighted" if to_weight else "unweighted"
        train_performances[weight_str] = []
        for train_size in train_range:
            stat = {}
            prediction_path = os.path.join(data_directory, "Train", "predictions-test-{}-{}".format(
                weight_str, train_size))
            evaluater = PartialEvaluater(test_path, prediction_path, group_id=group_id)
            stat = get_stat(evaluater)
            stat["train_size"] = train_size
            train_performances[weight_str].append(stat)

    train_performances['lambda'] = []
    for train_size in train_range:
        stat = {}
        prediction_path = os.path.join(data_directory, "lambdarank", "prediction-{}.txt".format(train_size))
        test_path = os.path.join(data_directory, "lambdarank", "test-{}.txt".format(train_size))
        evaluater = PartialEvaluater(test_path, prediction_path, group_id=group_id)
        stat = get_stat(evaluater)
        stat["train_size"] = train_size
        train_performances['lambda'].append(stat)

    plot_curve(train_performances, 'Avg. Rank', train_range, {'Full-info': full_performance['Avg. Rank']},
               'avg_rank.pdf')
    plot_curve(train_performances, 'DCG', train_range, {'Full-info': full_performance['DCG']}, 'dcg.pdf')
    plot_curve(train_performances, 'disparity', train_range, {'Full-info': full_performance['disparity']},
               'disparity.pdf')
    plot_curve(train_performances, 'rel_exposure', train_range, {'Full-info': full_performance['rel_exposure']},
               'rel_exposure.pdf')
