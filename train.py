import numpy as np
import math
import random
import copy
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models import convert_vars_to_gpu
from utils import logsumexp, shuffle_combined, exp_lr_scheduler, get_optimizer, serialize, transform_dataset
from evaluation import compute_dcg_rankings, evaluate_model, multiple_sample_and_log_probability

from fairness_loss import GroupFairnessLoss, BaselineAshudeepGroupFairnessLoss, get_group_merits, get_group_identities


def log_and_print(model,
                  data_reader,
                  writer: SummaryWriter,
                  step,
                  name="val",
                  experiment_name=None,
                  gpu=True,
                  fairness_evaluation=False,
                  exposure_relevance_plot=False,
                  deterministic=True,
                  group_fairness_evaluation=False,
                  args=None):
    position_bias_vector = 1. / torch.arange(1.,
                                             100.) ** args.position_bias_power
    if gpu:
        position_bias_vector = position_bias_vector.cuda()
    results = evaluate_model(
        model,
        data_reader,
        deterministic=deterministic,
        fairness_evaluation=fairness_evaluation,
        num_sample_per_query=args.sample_size,
        # position_bias_vector=1. / np.log2(2 + np.arange(200)),
        position_bias_vector=position_bias_vector,
        group_fairness_evaluation=group_fairness_evaluation,
        track_other_disparities=args.track_other_disparities,
        args=args)
    """
    Evaluate
    """
    if group_fairness_evaluation:
        avg_group_exposure_disparity, avg_group_asym_disparity = results[
                                                                     "avg_group_disparity"], results[
                                                                     "avg_group_asym_disparity"]
        if args.track_other_disparities:
            other_disparities = results["other_disparities"]

    avg_ndcg, avg_dcg, average_rank = results["ndcg"], results["dcg"], results["avg_rank"]
    """
    Return
    """
    returned = args.lambda_reward * avg_dcg
    if args.lambda_group_fairness > 0:
        returned -= args.lambda_group_fairness * avg_group_asym_disparity
    """
    Print
    """
    curve_pre_text = "{}_{}".format(name, args.fullinfo)
    print("Step {}, Average {}: NDCG: {}, DCG {}, Average Rank {}".
          format(step, curve_pre_text, avg_ndcg, avg_dcg, average_rank))
    if group_fairness_evaluation:
        print(
            "Average {} Group Exposure disparity: {}, Group Asymmetric disparity: {}".
                format(curve_pre_text, avg_group_exposure_disparity,
                       avg_group_asym_disparity, avg_group_asym_disparity))

    """
    Log
    """
    if experiment_name is None:
        experiment_name = "/"
    else:
        experiment_name += "/"
    if writer is not None:
        writer.add_scalars(experiment_name + "ndcg",
                           {curve_pre_text: avg_ndcg}, step)
        writer.add_scalars(experiment_name + "rank",
                           {curve_pre_text: average_rank}, step)
        writer.add_scalars(experiment_name + "dcg",
                           {curve_pre_text: avg_dcg}, step)
        writer.add_scalars(experiment_name + "metric",
                           {curve_pre_text: returned}, step)
        if group_fairness_evaluation:
            writer.add_scalars(experiment_name + "avg_group_disparity", {
                curve_pre_text:
                    avg_group_exposure_disparity
            }, step)
            writer.add_scalars(experiment_name + "avg_group_asym_disparity", {
                curve_pre_text:
                    avg_group_asym_disparity
            }, step)
            if args.track_other_disparities:
                for k, v in other_disparities.items():
                    writer.add_scalars(
                        experiment_name + "avg_group_asym_disparity",
                        {curve_pre_text + "_" + k: v[0]},
                        step)
                    writer.add_scalars(
                        experiment_name + "avg_group_disparity",
                        {curve_pre_text + "_" + k: v[1]},
                        step)

        # log on the train_dcg graph if evaluating on other training set
        if "_train--TRAIN" in name:
            writer.add_scalars(experiment_name + "train_dcg",
                               {curve_pre_text: avg_dcg}, step)
            writer.add_scalars(experiment_name + "train_ndcg",
                               {curve_pre_text: avg_ndcg}, step)

    return returned


def on_policy_training(data_reader,
                       validation_data_reader,
                       model,
                       writer=None,
                       experiment_name=None,
                       args=None):
    other_str = "full" if args.fullinfo == "partial" else "partial"
    position_bias_vector = 1. / torch.arange(1.,
                                             100.) ** args.position_bias_power
    lr = args.lr
    num_epochs = args.epochs
    weight_decay = args.weight_decay
    sample_size = args.sample_size
    entropy_regularizer = args.entropy_regularizer

    print("Starting training with the following config")
    print(
        "Batch size {}, Learning rate {}, Weight decay {}, Entropy Regularizer {}, Entreg Decay {} Sample size {}\n"
        "Lambda_reward: {}, lambda_ind_fairness:{}, lambda_group_fairness:{}".
            format(args.batch_size, lr, weight_decay, args.entropy_regularizer,
                   args.entreg_decay, sample_size,
                   args.lambda_reward, args.lambda_ind_fairness,
                   args.lambda_group_fairness))

    if args.gpu:
        print("Use GPU")
        model = model.cuda()
        position_bias_vector = position_bias_vector.cuda()

    optimizer = get_optimizer(model.parameters(), lr, args.optimizer,
                              weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_decay, min_lr=1e-6, verbose=True,
        patience=6)

    train_feats, train_rels = data_reader
    train_dataset = torch.utils.data.TensorDataset(train_feats, train_rels)
    valid_feats, valid_rels = validation_data_reader
    len_train_set = len(train_feats) // args.batch_size + 1
    fairness_evaluation = True if args.lambda_ind_fairness > 0.0 else False
    group_fairness_evaluation = True

    if group_fairness_evaluation and args.disparity_type != 'ashudeep':
        with torch.no_grad():
            group0_merit, group1_merit = get_group_merits(
                train_feats, train_rels, args.group_feat_id, args.group_feat_threshold, mean=False)
            print("Group 0 mean merit: {}, Group1 mean merit: {}".format(
                group0_merit, group1_merit))
            sign = 1.0 if group0_merit >= group1_merit else -1.0
            if args.disparity_type != 'ashudeep_mod':
                # random starting estimate for group_disparity indicator
                group_disparity_indicator_batch_size = args.group_disparity_indicator_batch_size * args.batch_size
                if group_disparity_indicator_batch_size > 4000:
                    group_disparity_indicator_batch_size = 4000
                if group_disparity_indicator_batch_size < 1000:
                    group_disparity_indicator_batch_size = 1000
                rand_ids = random.choices(
                    range(len(train_rels)), k=group_disparity_indicator_batch_size)
                group_disp_feats = train_feats[rand_ids]
                group_disp_rels = train_rels[rand_ids]
                if args.gpu:
                    group_disp_feats, group_disp_rels = group_disp_feats.cuda(), group_disp_rels.cuda()
                indicator_dataset = torch.utils.data.TensorDataset(group_disp_feats, group_disp_rels)
                indicator_dataloader = torch.utils.data.DataLoader(indicator_dataset, batch_size=args.batch_size,
                                                                   shuffle=True)
                indicator_disparities = []
                for data in indicator_dataloader:
                    feats, rel = data
                    scores = model(feats).squeeze(-1)
                    rankings = multiple_sample_and_log_probability(
                        scores, sample_size, return_prob=False, batch=True)
                    group_identities = get_group_identities(feats, args.group_feat_id, args.group_feat_threshold)
                    indicator_disparity = GroupFairnessLoss.compute_multiple_group_disparity(rankings, rel,
                                                                                             group_identities,
                                                                                             group0_merit,
                                                                                             group1_merit,
                                                                                             position_bias_vector,
                                                                                             args.disparity_type,
                                                                                             noise=args.noise,
                                                                                             en=args.en).mean(dim=-1)
                    indicator_disparities.append(indicator_disparity)
                indicator_disparities = torch.cat(indicator_disparities, dim=0)
                print("Disparities indicator: {}".format(indicator_disparities.mean().item()))

    if args.early_stopping:
        time_since_best = 0
        best_metric = -1e6
        best_model = None
        best_epoch = None

    entropy_list = []
    sum_loss_list = []
    rewards_list = []
    fairness_loss_list = []
    reward_variance_list = []
    train_ndcg_list = []
    train_dcg_list = []
    weight_list = []

    epoch_iterator = range(num_epochs)

    for epoch in epoch_iterator:

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        if args.progressbar:
            train_dataloader = tqdm(train_dataloader)

        for batch_id, data in enumerate(train_dataloader):
            feats, rel = data

            scores = model(feats).squeeze(-1)
            probs = nn.functional.softmax(scores, dim=-1)
            rankings, log_model_prob = multiple_sample_and_log_probability(
                scores, sample_size, return_prob=True, batch=True)

            with torch.no_grad():
                ndcgs, dcgs = compute_dcg_rankings(rankings, rel)
                utility_list = ndcgs if args.reward_type == "ndcg" else dcgs
                # FAIRNESS constraints
                if args.lambda_group_fairness > 0.0:
                    if args.unweighted_fairness:
                        rel = (rel > 0.0).float()
                    group_identities = get_group_identities(
                        feats, args.group_feat_id, args.group_feat_threshold)
                    if args.disparity_type == "ashudeep_mod":
                        group_fairness_coeffs = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector, sign=sign)
                    elif args.disparity_type == "ashudeep":
                        group_fairness_coeffs = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector)
                    else:
                        indicator_disparities, group_fairness_coeffs = GroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities,
                            position_bias_vector,
                            group0_merit,
                            group1_merit,
                            indicator_disparities,
                            args.disparity_type,
                            indicator_type=args.indicator_type,
                            noise=args.noise,
                            en=args.en)
            optimizer.zero_grad()

            if args.lambda_group_fairness != 0.0:
                rewards = args.lambda_reward * utility_list - \
                          args.lambda_group_fairness * group_fairness_coeffs
            else:
                rewards = args.lambda_reward * utility_list
            rewards = rewards / (args.lambda_reward + args.lambda_group_fairness)
            baseline = 0.0
            if args.use_baseline:
                if args.baseline_type == "value":
                    baseline = rewards.mean(dim=-1, keepdim=True)
                else:
                    raise NotImplementedError
            reinforce_loss = ((rewards - baseline) * (-log_model_prob)).mean()

            entropy_loss = 0.0
            entropy = get_entropy(probs).mean()
            if args.entropy_regularizer > 0.0:
                entropy_loss = entropy_regularizer * (-entropy)

            sum_loss = reinforce_loss + entropy_loss
            sum_loss.backward()
            optimizer.step()
            # log the reward/dcg variance
            sum_loss_list.append(sum_loss.item())
            if args.lambda_group_fairness != 0.0:
                fairness_loss_list.append(group_fairness_coeffs.mean().item())
            reward_variance_list.append(utility_list.var(dim=1).mean().item())
            rewards_list.append(utility_list.mean().item())
            entropy_list.append(entropy.item())
            train_ndcg_list.append(ndcgs.mean(dim=1).sum().item())
            train_dcg_list.append(dcgs.mean(dim=1).sum().item())
            weight_list.append(rel.sum().item())

            step = epoch * len_train_set + batch_id

            if step % args.write_losses_interval == 0 and step > 0:
                """
                    LOGGING
                """
                weight_sum = np.sum(weight_list)
                log_output = "\nAverages of last 1000 rewards: {}, ndcgs: {}, dcgs: {}".format(
                    np.mean(rewards_list),
                    np.mean(train_ndcg_list),
                    np.sum(train_dcg_list) / weight_sum)
                if args.lambda_group_fairness > 0.0:
                    log_output += " disparity: {}".format(
                        np.mean(fairness_loss_list))
                print(log_output)
                if writer is not None:
                    writer.add_scalars(experiment_name + "/{}_sum_train_loss".format(
                        args.fullinfo), {"sum_loss": np.mean(sum_loss_list)}, step)
                    writer.add_scalars(
                        experiment_name + "/{}_var_reward".format(args.fullinfo),
                        {"var_reward": np.mean(reward_variance_list)}, step)
                    writer.add_scalars(
                        experiment_name + "/{}_entropy".format(args.fullinfo),
                        {"entropy": np.mean(entropy_list)}, step)
                    if args.lambda_group_fairness != 0.0:
                        writer.add_scalars(experiment_name + "/{}_fairness_loss".format(
                            args.fullinfo), {"fairness_loss": np.mean(fairness_loss_list)}, step)
                    writer.add_scalars(
                        experiment_name + "/{}_train_ndcg".format(args.fullinfo),
                        {"train_ndcg": np.mean(train_ndcg_list)}, step)
                    writer.add_scalars(
                        experiment_name + "/{}_train_dcg".format(args.fullinfo),
                        {"train_dcg": np.sum(train_dcg_list) / np.sum(weight_list)}, step)
                fairness_loss_list = []
                reward_variance_list = []
                sum_loss_list = []
                entropy_list = []
                weight_list = []
                train_ndcg_list = []
                train_dcg_list = []

            if step % args.evaluate_interval == 0 and step > 0:
                print(
                    "Evaluating on validation set: iteration {}/{} of epoch {}".
                        format(batch_id, len_train_set, epoch))
                curr_metric = log_and_print(
                    model,
                    (valid_feats, valid_rels),
                    writer,
                    step,
                    "TEST_full--TRAIN",
                    experiment_name,
                    args.gpu,
                    fairness_evaluation=fairness_evaluation,
                    # exposure_relevance_plot=exposure_relevance_plot,
                    deterministic=args.validation_deterministic,
                    group_fairness_evaluation=group_fairness_evaluation,
                    args=args)

                # LR and Entropy decay
                scheduler.step(curr_metric)
                # """
                # Early stopping
                # """
                if args.early_stopping:
                    if best_model is None or curr_metric > best_metric + abs(best_metric) * 0.0001:
                        best_metric = curr_metric
                        best_model = copy.deepcopy(model)
                        best_epoch = epoch
                        time_since_best = 0
                    else:
                        time_since_best += 1
                    if time_since_best >= 3:
                        entropy_regularizer = args.entreg_decay * entropy_regularizer
                        print("Decay entropy regularizer to {}".format(entropy_regularizer))
                    if time_since_best >= args.stop_patience:
                        print(
                            "Validation set metric hasn't increased in 10 steps. Exiting")
                        return best_model, best_metric

    return model, curr_metric


def get_entropy(probs):
    return -torch.sum(torch.log(probs + 1e-10) * probs, dim=-1)


def compute_baseline(state, type="max"):
    if type == "max":
        print("Depracated: Doesn't work anymore")
        rel = state
        max_dcg = 0.0
        for i in range(sum(rel)):
            max_dcg += 1.0 / math.log(2 + i)
        return max_dcg
    elif type == "value":
        rankings, rewards_list = state
        # state is sent as a set of rankings sampled using the policy and
        # the set of relevant documents
        return np.mean(rewards_list)
    else:
        print("-----No valid reward type selected-------")


def compute_multiple_log_model_probability(scores, rankings, gpu=None):
    subtracts = torch.zeros_like(rankings, dtype=torch.float)
    log_probs = torch.zeros_like(rankings, dtype=torch.float)
    batch_index = torch.arange(rankings.size()[0])
    scores = scores.squeeze(-1)
    if gpu:
        subtracts, log_probs = convert_vars_to_gpu([subtracts, log_probs])
        batch_index = convert_vars_to_gpu(batch_index)
    for j in range(rankings.size()[1]):
        posj = rankings[:, j]
        log_probs[:, j] = scores[posj] - logsumexp(scores - subtracts, dim=1)
        subtracts[batch_index, posj] = scores[posj] + 1e6
    return torch.sum(log_probs, dim=1)


def compute_log_model_probability(scores, ranking, gpu=None):
    """
    more stable version
    if rel is provided, use it to calculate probability only till
    all the relevant documents are found in the ranking
    """
    subtracts = torch.zeros_like(scores)
    log_probs = torch.zeros_like(scores)
    if gpu:
        subtracts, log_probs = convert_vars_to_gpu([subtracts, log_probs])
    for j in range(scores.size()[0]):
        posj = ranking[j]
        log_probs[j] = scores[posj] - logsumexp(scores - subtracts, dim=0)
        subtracts[posj] = scores[posj] + 1e6
    return torch.sum(log_probs)
