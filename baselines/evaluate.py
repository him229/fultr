import math
import statistics
from collections import defaultdict
from baseline import read_rank_dataset


def dcg_rank(rank):
    return 1 / math.log(2 + rank, 2)


def average_rank(rank):
    return rank


def reciprocal_rank(rank):
    return 1 / (rank + 1)


def wta_rank(rank):
    return int(rank == 0)


class PartialEvaluater:
    def __init__(self, test_file, predict_file, group_id=None, group_threshold=None):
        self.queries = {}
        if group_id is not None:
            self.groups = set()
        self.group_threshold = group_threshold
        with open(predict_file) as predict:
            for data, predict_line in zip(read_rank_dataset(test_file), predict):
                label, qid, features, cost = data
                if qid not in self.queries:
                    self.queries[qid] = {'scores': [], 'rel_docs': {}, 'groups': []}
                doc_id = len(self.queries[qid]['scores'])
                self.queries[qid]['scores'].append(float(predict_line.strip()))
                if int(label) != 0:
                    self.queries[qid]['rel_docs'][doc_id] = cost
                if group_id is not None:
                    group_feature = features[group_id + 1]
                    self.queries[qid]['groups'].append(group_feature)
                    if group_threshold is None:
                        self.groups.add(group_feature)
        if group_id is not None:
            self.groups = list(self.groups)

    @staticmethod
    def disparity1(m0, e0, m1, e1, global_m0, global_m1):
        return e0 / global_m0 - e1 / global_m1

    @staticmethod
    def disparity3(m0, e0, m1, e1, global_m0=None, global_m1=None):
        return e0 * m1 - e1 * m0

    @staticmethod
    def disparity2(m0, e0, m1, e1, global_m0=None, global_m1=None):
        return m1 / e1 - m0 / e0

    def compute_group_disparity(self, disparity_type='ratio'):
        if self.group_threshold is None:
            assert len(self.groups) == 2
            group0, group1 = self.groups[0], self.groups[1]
        else:
            group0, group1 = 0.0, 1.0
        group_mean_merits = self.compute_group_merit(mean=False)
        group_merits = self.compute_group_merit(mean=False)
        group0_merit, group1_merit = group_merits[group0], group_merits[group1]
        group_disparities = []
        if disparity_type == 'disp2':
            disparity_function = self.disparity2
        elif disparity_type == 'disp3':
            disparity_function = self.disparity3
        elif disparity_type == 'disp1':
            disparity_function = self.disparity1
        else:
            raise NotImplementedError
        for query in self.queries.values():
            group_merit, group_exposure = defaultdict(list), defaultdict(list)
            ranked = sorted(range(len(query['scores'])), key=lambda x: query['scores'][x], reverse=True)
            for rank, doc_id in enumerate(ranked):
                group = query['groups'][doc_id]
                if self.group_threshold is not None:
                    group = float(group > self.group_threshold)
                group_merit[group].append(query['rel_docs'].get(doc_id, 0.0))
                group_exposure[group].append(1 / (rank + 1))
            if len(group_exposure) > 1:
                disparity = disparity_function(sum(group_merit[group0]),
                                               sum(group_exposure[group0]),
                                               sum(group_merit[group1]),
                                               sum(group_exposure[group1]),
                                               group0_merit,
                                               group1_merit)
                group_disparities.append(disparity)
        return statistics.mean(group_disparities) ** 2

    def compute_group_merit(self, mean=True):
        if self.group_threshold is None:
            group_merit = {group: [] for group in self.groups}
        else:
            group_merit = {0.0: [], 1.0: []}
        for query in self.queries.values():
            group_nums, group_rels = defaultdict(int), defaultdict(int)
            for doc_id, group in enumerate(query['groups']):
                if self.group_threshold is not None:
                    group = float(group > self.group_threshold)
                group_nums[group] += 1
            for rel_doc, weight in query['rel_docs'].items():
                group = query['groups'][rel_doc]
                if self.group_threshold is not None:
                    group = float(group > self.group_threshold)
                group_rels[group] += weight
            if len(group_nums) > 1:
                if mean:
                    for group, group_num in group_nums.items():
                        group_merit[group].append(group_rels[group] / group_num)
                else:
                    for group, group_num in group_nums.items():
                        group_merit[group].append(group_rels[group])
        return {key: statistics.mean(value) for key, value in group_merit.items()}

    def compute_group_exposure(self, relevance=True):
        # TODO This function is not weighted yet
        if self.group_threshold is None:
            group_exposures = {group: 0.0 for group in self.groups}
            group_weights = {group: 0 for group in self.groups}
        else:
            group_exposures = {0.0: 0.0, 1.0: 0.0}
            group_weights = {0.0: 0.0, 1.0: 0.0}
        for query in self.queries.values():
            ranked = sorted(range(len(query['scores'])), key=lambda x: query['scores'][x], reverse=True)
            for rank, doc_id in enumerate(ranked):
                if doc_id in query['rel_docs'] or not relevance:
                    group = query['groups'][doc_id]
                    weight = query['rel_docs'].get(doc_id, 1.0)
                    if self.group_threshold is not None:
                        group = float(group > self.group_threshold)
                    group_exposures[group] += 1 / (rank + 1) * weight
                    group_weights[group] += weight
        return {group: group_exposures[group] / weight for group, weight in group_weights.items() if weight > 0}

    def rank(self):
        ranked_results = {}
        for qid, query in self.queries.items():
            ranked = sorted(range(len(query['scores'])), key=lambda x: query['scores'][x], reverse=True)
            ranked_results[qid] = ranked
        return ranked_results

    def full_evaluate(self, rank_function, to_normalize=False, topk=None):
        sums, num = 0.0, 0
        for query in self.queries.values():
            rel_docs = query['rel_docs']
            if not rel_docs:
                continue
            scores = query['scores']
            value, normalize = 0.0, 0.0
            ranked = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
            current = 0
            for rel_doc in rel_docs:
                rank = ranked.index(rel_doc)
                if topk is None or rank < topk:
                    value += rank_function(rank)
                if topk is None or current < topk:
                    normalize += rank_function(current)
                current += 1
            num += 1
            if to_normalize:
                sums += value / normalize
            else:
                sums += value
        return sums / len(self.queries)

    def partial_evaluate(self, rank_function):
        normalizer = 0.0
        sum = 0.0
        for query in self.queries.values():
            scores = query['scores']
            rel_docs = query['rel_docs']
            for rel_doc in rel_docs:
                weight = rel_docs[rel_doc]
                ranked = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
                rank = ranked.index(rel_doc)
                sum += rank_function(rank) * weight
                normalizer += weight
        return sum / normalizer
