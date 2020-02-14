import numpy as np
import matplotlib.pyplot as plt
from evaluate import *
import os
import matplotlib
from utils import unserialize

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

svmrank_directory = "../svm_proprank"
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