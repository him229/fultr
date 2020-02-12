#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import os
import time
from tensorboardX import SummaryWriter
import ast

from parse_args import args
from datareader import reader_from_pickle
from train import on_policy_training
from models import LinearModel, MLP
from evaluation import evaluate_model
from utils import serialize, transform_dataset

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    test_data = reader_from_pickle(args.full_test_data)

    torch.set_num_threads(args.num_cores)

    weighted_validation_data_reader = reader_from_pickle(
        args.eval_weighted_val_location) if args.eval_weighted_val else None

    other_train_data_reader = reader_from_pickle(
        args.eval_other_train_location) if args.eval_other_train else None

    model = torch.load(args.load_model)
    if args.mask_group_feat:
        model.masked_feat_id = args.mask_group_feat
    else:
        model.masked_feat_id = None
    print(list(model.named_parameters()))
    test_data = transform_dataset(test_data.data, args.gpu, True)
    results = evaluate_model(
        model, test_data, fairness_evaluation=False,
        group_fairness_evaluation=True, track_other_disparities=True,
        deterministic=args.evaluation_deterministic,
        args=args, num_sample_per_query=args.sample_size, normalize=True)
    print(results)
