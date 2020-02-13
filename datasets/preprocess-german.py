import numpy as np
import os
from sklearn.preprocessing import RobustScaler

from utils import serialize


def foo(attr, start, count):
    return ['{}{}'.format(attr, i) for i in range(start, count+1)]


def convert_to_categorical_data(a, new_data):
    a1 = foo('A1', 1, 4)
    a2 = [1]
    a3 = foo('A3', 0, 4)
    a4 = foo('A4', 0, 10)
    a5 = [4]
    a6 = foo('A6', 1, 5)
    a7 = foo('A7', 1, 5)
    a8 = [7]
    a9 = foo('A9', 1, 5)
    a10 = foo('A10', 1, 3)
    a11 = [10]
    a12 = foo('A12', 1, 4)
    a13 = [12]
    a14 = foo('A14', 1, 3)
    a15 = foo('A15', 1, 3)
    a16 = [15]
    a17 = foo('A17', 1, 4)
    a18 = [17]
    a19 = foo('A19', 1, 2)
    a20 = foo('A20', 1, 2)

    headers = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10+a11+a12+a13+a14+a15+a16+a17+a18+a19+a20

    for i, h in enumerate(headers):
        if not isinstance(h, int):
            new_data[:, i][np.where(a == h)[0]] = 1.
        else:
            new_data[:, i] = a[:, h].astype(float)

    # columns to be deleted
    del_list = []
    for i in range(new_data.shape[1]):
        if np.sum(new_data[:, i]) == 0:
            del_list.append(i)
    # delete the colums for gender and unused ones, keep one for rels
    delete_cols = set(list(range(33, 38)) + del_list[:-1])
    delete_cols

    headers_del = []
    for i in delete_cols:
        if i < len(headers):
            headers_del.append(i)
    headers_ = []
    for i, h in enumerate(headers):
        if i not in headers_del:
            headers_.append(headers[i])
    headers_.insert(4, "A9_Gender")
    headers_.append("RELEVANCE")

    males = np.concatenate(
        (np.where(a == 'A91')[0],
         np.where(a == 'A93')[0],
         np.where(a == 'A94')[0]))

    new_data_del = np.delete(new_data, list(delete_cols), 1)
    # add gender in the 4th column
    new_data_ins = np.insert(new_data_del, 4, np.zeros(1000), axis=1)
    new_data_ins[:, 4][males] = 1
    # add relevances in the last column
    new_data_ins[:, -1][a[:, -1].astype(float) == 1] = 1.0

    # ALTERNATE group feature
    # bad credit
    # g1 = np.concatenate((np.where(a == 'A30')[0], np.where(a == 'A31')[0], np.where(a == 'A32')[0]))
    # g0 = np.concatenate((np.where(a == 'A33')[0], np.where(a == 'A34')[0]))
    # group_col_num, other_del_cols = 6, [7,8,9,10]

    # car purpose
    g1 = np.concatenate(
        (np.where(a == 'A42')[0],
         np.where(a == 'A43')[0],
         np.where(a == 'A44')[0],
         np.where(a == 'A45')[0],
         np.where(a == 'A46')[0],
         np.where(a == 'A47')[0],
         np.where(a == 'A48')[0],
         np.where(a == 'A49')[0],
         np.where(a == 'A410')[0]))
    g0 = np.concatenate((np.where(a == 'A40')[0], np.where(a == 'A41')[0]))
    group_col_num, other_del_cols = 14, [12, 13, 14, 15, 16, 17, 18, 19, 20]
    assert len(g1) + len(g0) == new_data_ins.shape[0]

    new_data_ins_del = np.delete(new_data_ins, other_del_cols, 1)
    new_data_ins_del[:, group_col_num][g0] = 0
    new_data_ins_del[:, group_col_num][g1] = 1
    assert np.sum(new_data_ins_del[:, group_col_num]) == len(g1)
    # Changed the categorical features to binary and the group column is 11
    return new_data_ins_del

# np.savetxt('GermanCredit/prod/german_mod_car_11.data',
#            new_data_ins_del, fmt='%-2d')


# data = np.loadtxt('../GermanCredit/prod/german_mod.data')


def get_nonzero(vals):
    return np.nonzero(vals)[0]


def get_train_test_splits(idxs):
    count_tr = int(len(idxs) * 0.35)
    count_va = int(len(idxs) * 0.35) + count_tr
    rand_idxs = np.random.rand(idxs.shape[0]).argsort()
    train_idxs = idxs[rand_idxs[:count_tr]]
    val_idxs = idxs[rand_idxs[count_tr:count_va]]
    test_idxs = idxs[rand_idxs[count_va:]]
    return train_idxs, val_idxs, test_idxs


def get_ranking_rel(data_, t_p, t_n, rel_ratio, non_rel_ratio):
    data_rel = data_[np.random.choice(t_p, rel_ratio, replace=False)]
    data_non_rel = data_[np.random.choice(t_n, non_rel_ratio, replace=False)]
    ranking = np.vstack([data_rel, data_non_rel])
    rand_idxs = np.random.rand(ranking.shape[0]).argsort()
    return ranking[rand_idxs][:, :-1], ranking[rand_idxs][:, -1]


def get_final_rankings(
        data, data_, group_column_number, num_queries_tr, num_queries_va,
        num_queries_te, rel_ratio, save=False):
    non_rel_ratio = 20 - 2*rel_ratio
    train_rankings = []
    train_rels = []
    test_rankings = []
    test_rels = []
    val_rankings = []
    val_rels = []

    rels = data[:, -1]

    if group_column_number:
        # TODO - deal with validation set
        gender = data[:, group_column_number]
        l = []
        for g in [0., 1.]:
            for r in [0., 1.]:
                l.append(get_nonzero(np.logical_and(gender == g, rels == r)))
        gen_f_rel_n, gen_f_rel_p, gen_m_rel_n, gen_m_rel_p = l

        tr_gen_f_rel_n, va_gen_f_rel_n, te_gen_f_rel_n = get_train_test_splits(
            gen_f_rel_n)
        tr_gen_f_rel_p, va_gen_f_rel_p, te_gen_f_rel_p = get_train_test_splits(
            gen_f_rel_p)
        tr_gen_m_rel_n, va_gen_m_rel_n, te_gen_m_rel_n = get_train_test_splits(
            gen_m_rel_n)
        tr_gen_m_rel_p, va_gen_m_rel_p, te_gen_m_rel_p = get_train_test_splits(
            gen_m_rel_p)

        tr_p, va_p, te_p = np.hstack(
            [tr_gen_m_rel_p, tr_gen_f_rel_p]), np.hstack(
            [va_gen_f_rel_p, va_gen_m_rel_p]), np.hstack(
            [te_gen_m_rel_p, te_gen_f_rel_p])
        tr_n, va_n, te_n = np.hstack(
            [tr_gen_m_rel_n, tr_gen_f_rel_n]), np.hstack(
            [va_gen_f_rel_n, va_gen_m_rel_n]), np.hstack(
            [te_gen_m_rel_n, te_gen_f_rel_n])
    else:
        data_p = np.nonzero(rels == 1.)[0]
        data_n = np.nonzero(rels == 0.)[0]

        tr_p, va_p, te_p = get_train_test_splits(data_p)
        tr_n, va_n, te_n = get_train_test_splits(data_n)

    for i in range(num_queries_tr):
        ranking_train, rel_train = get_ranking_rel(
            data_, tr_p, tr_n, rel_ratio, non_rel_ratio)
        train_rankings.append(ranking_train)
        train_rels.append(rel_train)
    for i in range(num_queries_te):
        ranking_test, rel_test = get_ranking_rel(
            data_, te_p, te_n, rel_ratio, non_rel_ratio)
        test_rankings.append(ranking_test)
        test_rels.append(rel_test)
    for i in range(num_queries_va):
        ranking_val, rel_val = get_ranking_rel(
            data_, va_p, va_n, rel_ratio, non_rel_ratio)
        val_rankings.append(ranking_val)
        val_rels.append(rel_val)

    train_set = (train_rankings, train_rels)
    test_set = (test_rankings, test_rels)
    val_set = (val_rankings, val_rels)
    if save:
        base_path_full = os.path.join(base_path, "full")
        if not os.path.exists(base_path_full):
            os.makedirs(base_path_full)
        serialize(train_set, os.path.join(base_path_full, "train.pkl"))
        serialize(val_set, os.path.join(base_path_full, "valid.pkl"))
        serialize(test_set, os.path.join(base_path_full, "test.pkl"))
    return train_set, test_set


if __name__ == "__main__":
    base_path = "../transformed_datasets/german"
    base_path_raw = os.path.join(base_path, "raw")
    data_raw_path = os.path.join(base_path_raw, "german.data")

    a = np.loadtxt(data_raw_path, dtype='str')
    new_data = np.zeros((1000, 70))
    categorical_data = convert_to_categorical_data(a, new_data)

    data_ = RobustScaler(with_centering=False).fit_transform(categorical_data)
    group_column_number = 14
    np.set_printoptions(suppress=True)

    get_final_rankings(categorical_data, data_, group_column_number,
                       500, 500, 500, 1, save=True)
    print("Datasets successfully written")
