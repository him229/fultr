import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
from models import convert_vars_to_gpu

from fairness_loss import get_group_identities, get_group_merits, \
    GroupFairnessLoss, BaselineAshudeepGroupFairnessLoss


def sample_multiple_ranking(probs, sample_size):
    candidate_set_size = probs.shape[0]
    ranking = torch.multinomial(
        probs.expand(
            sample_size,
            -1
        ),
        num_samples=candidate_set_size,
        replacement=False
    )
    return ranking


def sample_ranking(probs, output_propensities=True):
    propensity = 1.0
    candidate_set_size = probs.shape[0]
    ranking = torch.multinomial(
        probs,
        num_samples=candidate_set_size,
        replacement=False
    )
    if output_propensities:
        for i in range(candidate_set_size):
            propensity *= probs[ranking[i]]
            probs[ranking[i]] = 0.0
            probs = probs / probs.sum()
        return ranking, propensity
    else:
        return ranking


def multiple_sample_and_log_probability(
        scores, sample_size, return_prob=True, batch=False):
    if not batch:
        assert scores.dim() == 1
        subtracts = scores.new_zeros((sample_size, scores.size(0)))
        batch_index = torch.arange(sample_size, device=scores.device)
        if return_prob:
            log_probs = torch.zeros_like(subtracts, dtype=torch.float)
        rankings = []
        for j in range(scores.size(0)):
            probs = nn.functional.softmax(scores - subtracts, dim=1) + 1e-10
            posj = torch.multinomial(probs, 1).squeeze(-1)
            rankings.append(posj)
            if return_prob:
                log_probs[:, j] = probs[batch_index, posj].log()
            subtracts[batch_index, posj] = scores[posj] + 1e6
        rankings = torch.stack(rankings, dim=1)
        if return_prob:
            log_probs = log_probs.sum(dim=1)
            return rankings, log_probs
        else:
            return rankings
    else:
        assert scores.dim() == 2
        batch_size, candidiate_size = scores.size(0), scores.size(1)
        subtracts = scores.new_zeros((batch_size, sample_size, candidiate_size))
        batch_index = torch.arange(
            batch_size, device=scores.device).unsqueeze(1).expand(
            batch_size, sample_size)
        sample_index = torch.arange(
            sample_size, device=scores.device).expand(
            batch_size, sample_size)
        if return_prob:
            log_probs = torch.zeros_like(subtracts, dtype=torch.float)
        rankings = []
        for j in range(scores.size(1)):
            probs = nn.functional.softmax(
                scores.unsqueeze(1) - subtracts, dim=-1) + 1e-10
            posj = torch.multinomial(
                probs.reshape(
                    batch_size * sample_size,
                    -1
                ),
                1
            ).squeeze(-1).reshape(batch_size, sample_size)
            rankings.append(posj)
            if return_prob:
                log_probs[:, :, j] = probs[batch_index,
                                           sample_index, posj].log()
            subtracts[batch_index, sample_index,
                      posj] = scores[batch_index, posj] + 1e6
        rankings = torch.stack(rankings, dim=-1)
        if return_prob:
            log_probs = log_probs.sum(dim=-1)
            return rankings, log_probs
        else:
            return rankings


def compute_average_rank(rankings,
                         relevance_vector,
                         relevance_threshold=0):
    relevant = (relevance_vector > relevance_threshold).float()
    document_ranks = rankings.sort(dim=-1)[1].float()
    avg_rank = (document_ranks * (relevance_vector * relevant).unsqueeze(1)).sum(dim=-1)
    return avg_rank


def compute_dcg(ranking, relevance_vector, k=None):
    N = len(relevance_vector)
    if k is None:
        k = N
    ranking = ranking[:min((k, N))]
    len_ranking = float(len(ranking))
    sorted_relevances = -torch.sort(-relevance_vector)[0][:min((k, N))]
    len_sorted_relevances = float(len(sorted_relevances))

    dcgmax = torch.sum(sorted_relevances / torch.log2(
        torch.arange(len_sorted_relevances) + 2).to(relevance_vector.device))
    dcg = torch.sum(relevance_vector[ranking] / torch.log2(
        torch.arange(len_ranking) + 2).to(relevance_vector.device))
    if dcgmax == 0:
        dcgmax = 1.0
    return dcg / dcgmax, dcg


def compute_dcg_rankings(
        t_rankings, relevance_vector, binary_rel=False):
    """
    input t_rankings = [num_rankings X cand_set_size]
    returns dcg as a tensor of [num_rankings X 1] i.e. dcg for each ranking
    """
    if binary_rel:
        relevance_vector = (relevance_vector > 0).float()
    # t_rankings = t_rankings[:min((k, N)),:]
    len_rankings = float(t_rankings.shape[-1])
    sorted_relevances = torch.sort(
        relevance_vector,
        dim=-1,
        descending=True
    )[0]
    dcg = torch.zeros_like(t_rankings, dtype=torch.float)
    dcg.scatter_(-1, t_rankings,
                 1.0 / torch.log2(torch.arange(len_rankings, device=relevance_vector.device) + 2).expand_as(t_rankings))
    dcg *= relevance_vector.unsqueeze(-2)
    dcg = dcg.sum(dim=-1)
    dcgmax = torch.sum(sorted_relevances * torch.log2(torch.arange(len_rankings, device=relevance_vector.device) + 2),
                       dim=-1)
    nonzero = (dcgmax != 0.0)
    ndcg = dcg.clone()
    ndcg[nonzero] /= dcgmax[nonzero].unsqueeze(-1)
    return ndcg, dcg


def get_relative_gain(relevance):
    return (2.0 ** relevance - 1) / 16


def compute_err(ranking, relevance_vector):
    """
    Defined in Chapelle 11a (Section 5.1.1)
    """
    err = 0.0
    for i, doc in enumerate(ranking):
        not_found_probability = 1.0
        for j in range(i):
            not_found_probability *= 1 - get_relative_gain(
                relevance_vector[ranking[j]])
        err += get_relative_gain(
            relevance_vector[doc]) * not_found_probability / (1 + i)
    return err


def pairwise_mse(exposures, relevances):
    mse = 0.0
    e_by_r = exposures / relevances
    N = len(relevances)
    for i in range(N):
        for j in range(i, N):
            mse += (e_by_r[i] - e_by_r[j]) ** 2
    return mse / (N * N)


def scale_invariant_mse(exposures, relevances):
    """
    https://arxiv.org/pdf/1406.2283v1.pdf Equation 1, 2, 3
    """
    # sqrt(Eq. 3)
    assert (np.all(
        np.isfinite(exposures) & np.isfinite(relevances) & (exposures > 0) &
        (relevances > 0)))
    log_diff = np.log(exposures) - np.log(relevances)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(
            np.sum(np.square(log_diff)) / num_pixels -
            np.square(np.sum(log_diff)) / np.square(num_pixels))


def asymmetric_disparity(exposures, relevances):
    disparities = []
    for i in range(len(exposures)):
        for j in range(len(exposures)):
            if relevances[i] >= relevances[j]:
                if relevances[j] > 0.0:
                    disparities.append(
                        max([
                            0, exposures[i] / relevances[i] -
                            exposures[j] / relevances[j]
                        ]))
                else:
                    disparities.append(0)
    if np.isnan(np.mean(disparities)):
        print("NAN occured at", exposures, relevances, disparities)
    return np.mean(disparities)


def evaluate_model(model,
                   validation_data,
                   num_sample_per_query=10,
                   deterministic=False,
                   fairness_evaluation=False,
                   position_bias_vector=None,
                   group_fairness_evaluation=False,
                   track_other_disparities=False,
                   args=None,
                   normalize=False,
                   noise=None,
                   en=None):
    if noise is None:
        noise = args.noise
    if en is None:
        en = args.en
    ndcg_list = []
    dcg_list = []
    rank_list = []
    weight_list = []
    if (fairness_evaluation
            or group_fairness_evaluation) and position_bias_vector is None:
        position_bias_vector = 1. / torch.arange(
            1., 100.) ** args.position_bias_power
        if args.gpu:
            position_bias_vector = position_bias_vector.cuda()

    val_feats, val_rel = validation_data

    all_exposures = []
    all_rels = []

    validation_dataset = torch.utils.data.TensorDataset(val_feats, val_rel)
    dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size)
    if args.progressbar:
        dataloader = tqdm(dataloader)

    if group_fairness_evaluation:
        if track_other_disparities:
            disparity_types = ['disp1', 'disp2', 'disp3', 'ashudeep', 'ashudeep_mod']
        else:
            disparity_types = [args.disparity_type]
        if 'disp2' in disparity_types or 'ashudeep_mod' in disparity_types:
            group0_merit, group1_merit = get_group_merits(
                val_feats,
                val_rel,
                args.group_feat_id,
                args.group_feat_threshold,
                mean=False
            )
            sign = 1.0 if group0_merit >= group1_merit else -1.0
        else:
            group0_merit, group1_merit = None, None
            sign = None
        group_disparities = {
            disparity_type: [] for disparity_type in disparity_types
        }
    model.eval()
    with torch.no_grad():
        for data in dataloader:  # for each query
            feats, rel = data
            scores = model(feats).squeeze(-1)
            scores = scores * args.eval_temperature
            if deterministic:
                num_sample_per_query = 1
                rankings = torch.sort(
                    scores,
                    descending=True, dim=-1)[1].unsqueeze(1)
            else:
                rankings = multiple_sample_and_log_probability(
                    scores,
                    num_sample_per_query,
                    return_prob=False,
                    batch=True
                )

            ndcgs, dcgs = compute_dcg_rankings(rankings, rel)
            rank = compute_average_rank(rankings, rel)
            dcg_list += dcgs.mean(dim=-1).tolist()
            ndcg_list += ndcgs.mean(dim=-1).tolist()
            rank_list += rank.mean(dim=-1).tolist()
            weight_list += rel.sum(dim=-1).tolist()

            if group_fairness_evaluation:
                group_identities = get_group_identities(
                    feats,
                    args.group_feat_id,
                    args.group_feat_threshold
                )
                inds_g0 = group_identities == 0
                inds_g1 = group_identities == 1

                if args.unweighted_fairness:
                    rel = (rel > 0.0).float()

                for disparity_type in disparity_types:
                    if disparity_type == 'ashudeep':
                        disparity = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector).mean(dim=-1)
                    elif disparity_type == 'ashudeep_mod':
                        disparity = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector, sign=sign).mean(
                            dim=-1)
                    else:
                        disparity = GroupFairnessLoss.compute_multiple_group_disparity(
                            rankings,
                            rel,
                            group_identities,
                            group0_merit,
                            group1_merit,
                            position_bias_vector,
                            disparity_type=disparity_type,
                            noise=noise,
                            en=en
                        ).mean(dim=-1)
                    for i in range(len(rankings)):
                        if inds_g0[i].any() and inds_g1[i].any():
                            group_disparities[disparity_type].append(disparity[i].item())

    model.train()
    avg_ndcg = np.mean(ndcg_list)
    if normalize:
        avg_dcg = np.sum(dcg_list) / np.sum(weight_list)
        avg_rank = np.sum(rank_list) / np.sum(weight_list)
    else:
        avg_dcg = np.mean(dcg_list)
        avg_rank = np.mean(rank_list)

    results = {
        "ndcg": avg_ndcg,
        "dcg": avg_dcg,
        "avg_rank": avg_rank
    }
    if group_fairness_evaluation:
        # convert lists in dictionary to np arrays
        for disparity_type in group_disparities:
            group_disparities[disparity_type] = np.mean(
                group_disparities[disparity_type])

        other_disparities = {}
        for k, v in group_disparities.items():
            if k == 'ashudeep' or k == 'ashudeep_mod':
                disparity = v
                asym_disparity = v
            else:
                if args.indicator_type == "square":
                    disparity = v
                    asym_disparity = v ** 2
                elif args.indicator_type == "sign":
                    disparity = v
                    asym_disparity = abs(v)
                elif args.indicator_type == "none":
                    disparity = v
                    asym_disparity = v
                else:
                    raise NotImplementedError
            if k == args.disparity_type:
                avg_group_exposure_disparity = disparity
                avg_group_asym_disparity = asym_disparity
            other_disparities[k] = [asym_disparity, disparity]

        results.update({
            "avg_group_disparity": avg_group_exposure_disparity,
            "avg_group_asym_disparity": avg_group_asym_disparity
        })
        if track_other_disparities:
            results.update({"other_disparities": other_disparities})

    return results


def add_tiny_noise(one_hot_rel):
    """
    used to add tiny noise to avoid warnings in linregress
    """
    if one_hot_rel.min() == one_hot_rel.max():
        one_hot_rel = one_hot_rel + np.random.random(len(one_hot_rel)) * 1e-20
    return one_hot_rel


def optimal_exposure(num_relevant, num_docs, position_bias_function):
    """
    returns the optimal exposure that a randomized policy can give for
    the given number of relevant documents
    """
    top_k_exposure = np.mean(
        [position_bias_function(i) for i in range(num_relevant)])
    remaining_exposure = np.mean(
        [position_bias_function(i) for i in range(num_relevant, num_docs)])
    optimal_exposure = [top_k_exposure
                        ] * num_relevant + [remaining_exposure] * (
        num_docs - num_relevant)
    return optimal_exposure
