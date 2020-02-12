from utils import unserialize
from argparse import ArgumentParser
from gurobipy import Model, GRB
from fairness_loss import get_group_identities, GroupFairnessLoss
from evaluation import compute_dcg, compute_average_rank
from utils import serialize
import numpy as np
import torch
import math
import time
import random
import os
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--test_data", type=str, required=True)
parser.add_argument("--prediction_file", type=str, required=True)
parser.add_argument("--theta", type=float, default=0.0)
parser.add_argument("--group_feat_id", type=int, required=True)
parser.add_argument("--group_feat_threshold", type=float, default=None)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--true_merit", action='store_true', default=False)


def ideal_dcg(predicted_merit):
    predicted_merit = torch.sort(predicted_merit, descending=True)[0]
    log_weight = (torch.arange(len(predicted_merit), dtype=torch.float) + 2).log2()
    return (predicted_merit / log_weight).sum().item()


def solve_fairness_ranking(group_identity, group_merits, group_exposures, predicted_merit, position_bias, theta=0.0):
    candidate_size = len(group_identity)
    predicted_merit = predicted_merit / predicted_merit.sum()
    position_bias = position_bias[:candidate_size] / position_bias.sum()
    m = Model("mip1")
    m.setParam("OutputFlag", 0)
    group_merits = group_merits[group_identity]
    group_exposures = group_exposures[group_identity]
    cost = (group_exposures.unsqueeze(1) + position_bias.unsqueeze(0) -
            (group_merits + predicted_merit).unsqueeze(1)).abs()
    cost = {(i, j): cost[i, j].item() for i in range(candidate_size) for j in range(candidate_size)}
    ranking = m.addVars(candidate_size, candidate_size, obj=cost, vtype=GRB.BINARY)
    m.addConstrs((ranking.sum(i, '*') == 1 for i in range(candidate_size)))
    m.addConstrs((ranking.sum('*', i) == 1 for i in range(candidate_size)))
    idcg = ideal_dcg(predicted_merit)
    log_weight = (torch.arange(candidate_size, dtype=torch.float) + 2).log2()
    dcg = predicted_merit.unsqueeze(1) / log_weight.unsqueeze(0)
    dcg = {(i, j): dcg[i, j].item() for i in range(candidate_size) for j in range(candidate_size)}
    m.addConstr(ranking.prod(dcg) >= theta * idcg)
    m.optimize()
    solution = m.getAttr('x', ranking)
    ranking_result = [0 for _ in range(candidate_size)]
    curr_dcg = 0.0
    for i in range(candidate_size):
        for j in range(candidate_size):
            if solution[i, j] > 0:
                group = group_identity[i]
                ranking_result[j] = i
                curr_dcg += predicted_merit[i] / math.log2(j + 2)
                break
    return ranking_result


if __name__ == "__main__":
    args = parser.parse_args()

    test_data = unserialize(args.test_data)
    feats, rels = test_data
    predicted_merits = []
    with open(args.prediction_file) as file:
        i, merit = 0, []
        for line in file:
            value = float(line.strip())
            merit.append(value)
            if len(merit) == len(rels[i]):
                predicted_merits.append(torch.tensor(merit))
                merit = []
                i += 1
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    feats = [torch.as_tensor(feat, dtype=torch.float) for feat in feats]
    rels = [torch.as_tensor(rel, dtype=torch.float) for rel in rels]
    position_bias = 1.0 / torch.arange(1, 10000, dtype=torch.float)
    group_identities = [get_group_identities(
        feat, args.group_feat_id, args.group_feat_threshold).long() for feat in feats]
    group_merits, group_exposures = torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])
    group_disparity, disparity_num = 0.0, 0
    dcgs, weight = 0.0, 0.0
    ndcgs = []
    query_ids = list(range(len(feats)))

    with open(os.path.join(args.output_dir, "rankings.txt"), "w") as output:
        for i in range(10):
            for query_id in query_ids:
                feat, rel = feats[query_id], rels[query_id]
                if args.true_merit:
                    predicted_merit = rel
                else:
                    predicted_merit = predicted_merits[query_id]
                group_identity = group_identities[query_id]
                candidate_size = len(feat)
                ranking_result = solve_fairness_ranking(group_identity, group_merits, group_exposures,
                                                        predicted_merit, position_bias, args.theta)
                for pos, doc in enumerate(ranking_result):
                    group = group_identity[doc]
                    group_merits[group] += predicted_merit[doc] / predicted_merit.sum()
                    group_exposures[group] += position_bias[pos] / position_bias[:candidate_size].sum()
                output.write(str(ranking_result) + "\n")
                ranking_result = torch.as_tensor(ranking_result, dtype=torch.long)
                ndcg, dcg = compute_dcg(ranking_result, rel)
                dcgs += dcg.item()
                ndcgs.append(ndcg.item())
                weight += rel.sum().item()
                if (group_identity == 0).any() and (group_identity == 1).any():
                    group_disparity += GroupFairnessLoss.compute_group_disparity(
                        ranking_result, rel, group_identity, None, None, position_bias, 'disp3').item()
                    disparity_num += 1
                if disparity_num % 1000 == 0:
                    print(group_disparity / disparity_num)
                    print(dcgs / weight)
                    print(group_merits, group_exposures)
    group_disparity = group_disparity / disparity_num
    print("DCG: {}, Disparity: {}".format(dcgs / weight, group_disparity))
    results = {
        "dcg": dcgs / weight,
        "ndcg": np.mean(ndcgs),
        "other_disparities": {
            "disp3": [group_disparity ** 2, group_disparity]
        }
    }
    serialize(results, os.path.join(args.output_dir, "plt_data_pl_{}.json".format(time.strftime("%m-%d-%H-%M"))),
              in_json=True)
