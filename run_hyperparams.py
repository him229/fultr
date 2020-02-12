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
from utils import serialize, transform_dataset, unserialize

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.set_num_threads(args.num_cores)

    train_data = reader_from_pickle(
        args.partial_train_data) if args.fullinfo == "partial" else reader_from_pickle(args.full_train_data)
    train_data = train_data.data
    train_data = transform_dataset(
        train_data, args.gpu, args.weighted)

    val_data = reader_from_pickle(
        args.partial_val_data) if args.fullinfo == "partial" else reader_from_pickle(args.full_val_data)
    val_data = val_data.data
    val_data = transform_dataset(
        val_data, args.gpu, args.weighted)

    test_data = reader_from_pickle(args.full_test_data)
    test_data = test_data.data

    if args.summary_writing:
        if not os.path.exists(args.log_dir):
            try:
                os.makedirs(args.log_dir)
            except FileExistsError:
                pass
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    model_params_list = []
    a = ast.literal_eval(args.lambda_list)
    lambdas_list = [float(c) for c in a]
    plt_data = []
    plt_data_dict = []

    if not os.path.exists(args.hyperparam_folder):
        os.makedirs(args.hyperparam_folder)

    for i, lgroup in enumerate(lambdas_list):
        args.lambda_reward = 1.0
        args.lambda_ind_fairness = 0.0
        args.lambda_group_fairness = lgroup

        wd = args.weight_decay
        er = args.entropy_regularizer

        lgroup_name = str(
            lgroup) if lgroup >= 0.01 else "{:.1e}".format(lgroup)
        experiment_name = "{}_{}_lambda{}_lr{}_wd{}_er{}_ed{}".format(
            args.experiment_prefix, args.fullinfo, lgroup_name, args.lr,
            args.weight_decay, args.entropy_regularizer, args.entreg_decay)

        model_kwargs = {'clamp': args.clamp}
        if args.mask_group_feat:
            model_kwargs['masked_feat_id'] = args.group_feat_id
        if args.hidden_layer is None:
            model = LinearModel(
                input_dim=args.input_dim, **model_kwargs)
        else:
            model = MLP(input_dim=args.input_dim,
                        hidden_layer=args.hidden_layer,
                        dropout=args.dropout, **model_kwargs)

        result = on_policy_training(
            train_data, val_data, model, writer=writer,
            experiment_name=experiment_name, args=args)
        model, performance = result
        print(model)
        print("Get best performance {} at weight decay {}, entropy_regularizer {}".format(
                    performance, wd, er))

        test_data = transform_dataset(test_data, args.gpu, True)
        results_test = evaluate_model(
            model, test_data, fairness_evaluation=False,
            group_fairness_evaluation=True, track_other_disparities=True,
            deterministic=args.evaluation_deterministic,
            args=args, num_sample_per_query=args.sample_size, normalize=True,
            noise=False, en=0.0)
        print("Best performance on valid set: {}".format(performance))
        out_dict = {'best_perf': performance, "test": results_test, 'args': vars(args)}
        if args.eval_weighted_val:
            weighted_validation_data_reader = reader_from_pickle(args.eval_weighted_val_location)
            weighted_validation_data = transform_dataset(weighted_validation_data_reader.data, args.gpu, True)
            results_validation = evaluate_model(model, weighted_validation_data, fairness_evaluation=False,
                                                group_fairness_evaluation=True, track_other_disparities=True,
                                                deterministic=args.evaluation_deterministic, args=args,
                                                num_sample_per_query=args.sample_size, normalize=True, noise=False,
                                                en=0.0)
            out_dict['valid'] = results_validation
        if args.eval_other_train:
            other_train_data_reader = reader_from_pickle(args.eval_other_train_location)
            other_train_data = transform_dataset(other_train_data_reader.data, args.gpu, True)
            results_train = evaluate_model(model, other_train_data, fairness_evaluation=False,
                                           group_fairness_evaluation=True, track_other_disparities=True,
                                           deterministic=args.evaluation_deterministic, args=args,
                                           num_sample_per_query=args.sample_size, normalize=True, noise=False,
                                           en=0.0)
            out_dict['train'] = results_train
        out_dict.update({
            "gf_lambda": lgroup,
            "weight_decay": wd,
            "entropy_regularizer": er,
            "early_stopping": args.early_stopping,
            "full_info": args.fullinfo,
            "learning_rate": args.lr,
            "performance": performance
        })
        plt_data_dict.append(out_dict)
        if args.save_checkpoints:
            torch.save(model, os.path.join(
                args.hyperparam_folder, "best_{}_{}_lr{}_wd{}_er{}_es{}.ckpt".format(
                    args.fullinfo, lgroup, args.lr, wd, er,
                    args.early_stopping)))
    serialize(
        plt_data_dict, os.path.join(
            args.hyperparam_folder, 'plt_data_pl_{}_{}_tune{}_{}.json'.format(
                lambdas_list, args.fullinfo, args.tuning,
                time.strftime("%m-%d-%H-%M"))),
        in_json=True)
    if writer is not None:
        writer.close()
