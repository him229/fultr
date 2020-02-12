import numpy as np
import torch


def relevant_indices_to_onehot(rel, num_docs):
    onehot = np.zeros(num_docs)
    for relevant_doc in rel:
        onehot[relevant_doc] = 1
    return onehot


def get_group_merits(
        features, rels, group_feat_id, group_feat_threshold, mean=True):
    group_identities = get_group_identities(
        features, group_feat_id, group_feat_threshold)
    group0_merits, group1_merits = [], []
    for i in range(len(features)):
        inds_g0 = group_identities[i] == 0
        inds_g1 = group_identities[i] == 1
        if inds_g0.any() and inds_g1.any():
            # it depends on using mean or sum here
            if mean:
                group0_merits.append(rels[i][inds_g0].mean().item())
                group1_merits.append(rels[i][inds_g1].mean().item())
            else:
                group0_merits.append(rels[i][inds_g0].sum().item())
                group1_merits.append(rels[i][inds_g1].sum().item())
    group0_merit = np.mean(group0_merits)
    group1_merit = np.mean(group1_merits)
    return group0_merit, group1_merit


def get_group_identities(features, group_feat_id, group_feat_threshold=None):
    group_identities = features.select(
        dim=-1,
        index=group_feat_id
    )
    if group_feat_threshold is not None:
        group_identities = (group_identities > group_feat_threshold).float()
    return group_identities


def get_exposures(ranking, position_bias_vector):
    num_docs = len(ranking)
    exposure = torch.zeros(
        num_docs,
        device=ranking.device
    )
    exposure[ranking] = position_bias_vector[:num_docs]
    return exposure


def get_multiple_exposures(rankings, position_bias_vector):
    num_docs = rankings.shape[-1]
    pb_matrix = position_bias_vector[:num_docs].expand_as(rankings)
    exposures = torch.zeros_like(rankings).float()
    exposures = exposures.scatter_(
        -1,
        rankings,
        pb_matrix
    )
    return exposures


def get_expected_exposures(rankings, position_bias_vector):
    exposures_inv = 1 / get_multiple_exposures(
        rankings,
        position_bias_vector
    )
    exp_exposure = exposures_inv.mean(dim=0)
    exp_exposure = exp_exposure / rankings.shape[0]
    return exp_exposure


class GroupFairnessLoss:
    @staticmethod
    def compute_group_disparity(ranking,
                                rel,
                                group_identities,
                                group0_merit,
                                group1_merit,
                                position_biases,
                                disparity_type,
                                noise=False,
                                en=0.0):
        inds_g0 = group_identities == 0
        inds_g1 = group_identities == 1
        # if there is only one group in rankings, return 0
        if inds_g0.all() or inds_g1.all():
            return torch.zeros(
                1,
                dtype=torch.float,
                device=ranking.device
            )
        exposures = get_exposures(
            ranking,
            position_biases
        )
        group_disparities = None
        if disparity_type == "disp1":
            group_disparities = torch.mean(
                exposures[inds_g0]) / group0_merit - torch.mean(
                exposures[inds_g1]) / group1_merit
        elif disparity_type == "disp2":
            group_disparities = torch.mean(rel[inds_g1]) / torch.mean(
                exposures[inds_g1]) - torch.mean(
                rel[inds_g0]) / torch.mean(exposures[inds_g0])
        elif disparity_type == "disp3":
            group_disparities = torch.sum(
                rel[inds_g1]) * torch.sum(
                exposures[inds_g0]) - torch.sum(
                rel[inds_g0]) * torch.sum(
                exposures[inds_g1])
            # adjust loss for the noise
            if noise:
                group_disparities -= en * (
                    inds_g1.sum() * torch.sum(
                        exposures[inds_g0]) - inds_g0.sum() * torch.sum(
                        exposures[inds_g1]))
        else:
            raise NotImplementedError
        return group_disparities

    @staticmethod
    def compute_multiple_group_disparity(rankings,
                                         rel,
                                         group_identities,
                                         group0_merit,
                                         group1_merit,
                                         position_biases,
                                         disparity_type,
                                         noise=False,
                                         en=0.0):
        inds_g0 = (group_identities == 0).float()
        inds_g1 = (group_identities == 1).float()
        exposures = get_multiple_exposures(rankings, position_biases)
        # if there is only one group in rankings, return 0
        exposures_g0 = exposures * inds_g0.unsqueeze(1)
        exposures_g1 = exposures * inds_g1.unsqueeze(1)
        if disparity_type == "disp1":
            ratio0 = torch.sum(exposures_g0, dim=-1) / group0_merit
            ratio1 = torch.sum(exposures_g1, dim=-1) / group1_merit
            group_disparities = ratio0 - ratio1
        elif disparity_type == "disp2":
            g0_merit = torch.sum(rel * inds_g0, dim=-1)
            exposures_g0 = torch.sum(exposures_g0, dim=-1)
            ratio0 = g0_merit.unsqueeze(-1) / exposures_g0
            g1_merit = torch.sum(rel * inds_g1, dim=-1)
            exposures_g1 = torch.sum(exposures_g1, dim=-1)
            ratio1 = g1_merit.unsqueeze(-1) / exposures_g1
            group_disparities = ratio1 - ratio0
        elif disparity_type == "disp3":
            g0_merit = torch.sum(rel * inds_g0, dim=-1)
            exposures_g0 = torch.sum(exposures_g0, dim=-1)
            g1_merit = torch.sum(rel * inds_g1, dim=-1)
            exposures_g1 = torch.sum(exposures_g1, dim=-1)
            group_disparities = g1_merit.unsqueeze(-1) * exposures_g0 - g0_merit.unsqueeze(-1) * exposures_g1
            if noise:
                group_disparities -= en * (
                    inds_g1.sum(dim=-1).unsqueeze(-1) * exposures_g0 - inds_g0.sum(dim=-1).unsqueeze(-1) * exposures_g1)
        else:
            raise NotImplementedError
            # adjust loss for the noise, this only works for disp3
        single_group = (inds_g0.sum(dim=-1) * inds_g1.sum(dim=-1)) == 0
        group_disparities[single_group, :] = 0.0
        return group_disparities

    @staticmethod
    def compute_group_fairness_coeffs_generic(rankings,
                                              rels,
                                              group_identities,
                                              position_biases,
                                              group0_merit,
                                              group1_merit,
                                              indicator_disparities,
                                              disparity_type,
                                              indicator_type="square",
                                              noise=False,
                                              en=0.0):
        """
        compute disparity and then compute the gradient coefficients for
        asymmetric group disaprity loss
        """
        # compute average r_i/v_i for each group,
        # then the group which has higher relevance
        batch_size = rankings.size(0)
        group_disparities = GroupFairnessLoss.compute_multiple_group_disparity(
            rankings,
            rels,
            group_identities,
            group0_merit,
            group1_merit,
            position_biases,
            disparity_type,
            noise=noise,
            en=en)
        # update the indicator batch for every ranking in a query
        indicator_disparities = torch.cat(
            (indicator_disparities[batch_size:], group_disparities.mean(dim=-1)))
        if indicator_type == "square":
            indicator = indicator_disparities.mean()
        elif indicator_type == "sign":
            indicator = indicator_disparities.mean().sign()
        elif indicator_type == "none":
            indicator = 1.0
        else:
            raise NotImplementedError
        return indicator_disparities, indicator * group_disparities


class BaselineAshudeepGroupFairnessLoss:
    """
    Singh, Ashudeep, and Thorsten Joachims. "Policy Learning for Fairness in Ranking.
    " arXiv preprint arXiv:1902.04056 (2019).
    """

    @staticmethod
    def compute_group_disparity(ranking,
                                rel,
                                group_identities,
                                position_biases,
                                skip_zero=False):
        exposures = get_exposures(ranking, position_biases)
        inds_g0 = group_identities == 0
        inds_g1 = group_identities == 1
        if inds_g0.all() or inds_g1.all():
            return torch.zeros(ranking.size()[0],
                               dtype=torch.float, device=ranking.device)
        if skip_zero:
            inds_g0 = inds_g0 * (rel != 0)
            inds_g1 = inds_g1 * (rel != 0)
        g0_merit = torch.sum(rel[inds_g0])
        g1_merit = torch.sum(rel[inds_g1])
        exposures_g0 = torch.sum(exposures[inds_g0])
        exposures_g1 = torch.sum(exposures[inds_g1])
        group_disparities = 0.0
        if not (g0_merit == 0.0 or g1_merit == 0.0):
            ratio0 = exposures_g0 / g0_merit
            ratio1 = exposures_g1 / g1_merit
            group_disparities += ratio0 - ratio1
        return group_disparities

    @staticmethod
    def compute_multiple_group_disparity(rankings,
                                         rel,
                                         group_identities,
                                         position_biases,
                                         skip_zero=False):
        inds_g0 = (group_identities == 0).float()
        inds_g1 = (group_identities == 1).float()
        if skip_zero:
            inds_g0 = inds_g0 * (rel != 0).float()
            inds_g1 = inds_g1 * (rel != 0).float()
        exposures = get_multiple_exposures(rankings, position_biases)
        exposures_g0 = torch.sum(exposures * inds_g0.unsqueeze(1), dim=-1)
        exposures_g1 = torch.sum(exposures * inds_g1.unsqueeze(1), dim=-1)
        g0_merit = torch.sum(rel * inds_g0, dim=-1)
        g1_merit = torch.sum(rel * inds_g1, dim=-1)
        ratio0 = exposures_g0 / g0_merit.unsqueeze(-1)
        ratio1 = exposures_g1 / g1_merit.unsqueeze(-1)
        group_disparities = ratio0 - ratio1
        single_group = (g0_merit * g1_merit) == 0
        group_disparities[single_group, :] = 0.0
        return group_disparities

    @staticmethod
    def compute_group_fairness_coeffs_generic(rankings,
                                              rels,
                                              group_identities,
                                              position_biases,
                                              sign=None):
        """
        compute disparity and then compute the gradient coefficients for
        asymmetric group disaprity loss
        """
        inds_g0 = (group_identities == 0).float()
        inds_g1 = (group_identities == 1).float()
        # use sign if passed in (baseline_ashudeep_mod)
        if sign is None:
            sign = torch.ones(rankings.size(0), dtype=torch.float, device=rankings.device)
            num_g0, num_g1 = inds_g0.sum(dim=-1), inds_g1.sum(dim=-1)
            num_g0[num_g0 == 0.0] += 1
            num_g1[num_g1 == 0.0] += 1
            g0_merit = torch.sum(rels * inds_g0, dim=-1) / num_g0
            g1_merit = torch.sum(rels * inds_g1, dim=-1) / num_g1
            sign.masked_fill_(g0_merit < g1_merit, -1)
        group_disparities = BaselineAshudeepGroupFairnessLoss.compute_multiple_group_disparity(
            rankings, rels, group_identities, position_biases)
        indicator = (sign * group_disparities.mean(dim=-1)) > 0
        return (sign * indicator.float()).unsqueeze(-1) * group_disparities
