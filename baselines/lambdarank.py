import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import math
import argparse
import copy

from utils import unserialize
from baseline import RankingDataset, save_svmprop_train

metric_name = "DCG"


def activation_method(name):
    """
    :param name: (str)
    :return: torch.nn.Module
    """
    name = name.lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    else:
        return nn.Sequential()


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, final_size=0, final_activation="none", normalization="batch_norm",
                 activation='relu'):
        """
        :param input_size:
        :param hidden_layers: [(unit_num, normalization, dropout_rate)]
        :param final_size:
        :param final_activation:
        """
        nn.Module.__init__(self)
        self.input_size = input_size
        fcs = []
        last_size = self.input_size
        for size, to_norm, dropout_rate in hidden_layers:
            linear = nn.Linear(last_size, size)
            linear.bias.data.fill_(0.0)
            fcs.append(linear)
            last_size = size
            if to_norm:
                if normalization == 'batch_norm':
                    fcs.append(nn.BatchNorm1d(last_size))
                elif normalization == 'layer_norm':
                    fcs.append(nn.LayerNorm(last_size))
            fcs.append(activation_method(activation))
            if dropout_rate > 0.0:
                fcs.append(nn.Dropout(dropout_rate))
        self.fc = nn.Sequential(*fcs)
        if final_size > 0:
            linear = nn.Linear(last_size, final_size)
            linear.bias.data.fill_(0.0)
            finals = [linear, activation_method(final_activation)]
        else:
            finals = []
        self.final_layer = nn.Sequential(*finals)

    def forward(self, x):
        out = self.fc(x)
        out = self.final_layer(out)
        return out


class LambdaRank(nn.Module):
    def __init__(self, mlp_config):
        super(LambdaRank, self).__init__()
        self.model = MLP(**mlp_config)

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def fit(self, x, *args, rel_num=1, **kwargs):
        batch_size, doc_num, feature_num = x.size()
        doc_scores = self.forward(x)
        batch_index = torch.arange(batch_size, device=doc_scores.device)
        (sorted_scores, sorted_idxs) = doc_scores.sort(dim=1, descending=True)
        doc_ranks = doc_scores.new_zeros(doc_scores.size())
        # The rank here starts from 1
        doc_ranks[batch_index.unsqueeze(-1).expand_as(doc_scores), sorted_idxs] = 1 + torch.arange(doc_num,
                                                                                                   dtype=torch.float,
                                                                                                   device=doc_scores.device).expand(
            batch_size, doc_num)
        dcg = (1 / (doc_ranks[:, 0] + 1).log2()).mean()
        score_diffs = doc_scores[:, :rel_num].unsqueeze(-1) - doc_scores[:, rel_num:].unsqueeze(1)
        exped = 1 / (1 + score_diffs.exp())
        dcg_diffs = (1 / (1 + doc_ranks[:, :rel_num]).log2()).unsqueeze(-1) - (
                1 / (1 + doc_ranks[:, rel_num:]).log2()).unsqueeze(1)
        lamb_updates = exped * dcg_diffs.abs()
        lambs = torch.zeros(batch_size, doc_num, device=doc_scores.device)
        lambs[:, :rel_num] -= lamb_updates.sum(dim=2)
        lambs[:, rel_num:] += lamb_updates.sum(dim=1)
        doc_scores.backward(lambs)
        return dcg.detach()


class Regression(nn.Module):
    def __init__(self, mlp_config):
        super(Regression, self).__init__()
        self.model = MLP(**mlp_config)

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def fit(self, x, y, **kwargs):
        doc_scores = self.forward(x)
        loss = nn.functional.mse_loss(doc_scores, y)
        loss.backward()
        return -loss.detach()


def validate(model, valid_dataloader, output_path=None, metric=metric_name):
    model.eval()
    if output_path is not None:
        output = open(output_path, "w")
    sums, num = 0.0, 0
    with torch.no_grad():
        for batch in valid_dataloader:
            x, y = batch
            values = model(x)
            if metric == "DCG":
                _, indexes = values.sort(dim=1, descending=True)
                indexes = indexes.tolist()
                for i in range(len(indexes)):
                    index = indexes[i]
                    rel_docs = y[i].nonzero().squeeze(1).tolist()
                    for rel_doc in rel_docs:
                        sums += 1 / math.log(index.index(rel_doc) + 2, 2)
                        num += 1
            elif metric == "square":
                loss = -nn.functional.mse_loss(values, y, reduction='none')
                loss = loss.mean(dim=-1).sum().item()
                sums += loss
                num += y.size(0)
            else:
                raise NotImplementedError
            if output_path is not None:
                values = values.cpu().tolist()
                for value in values:
                    for v in value:
                        output.write(str(v) + "\n")
    model.train()
    return sums / num


def train(model, optimizer, train_dataset, epoch_num, valid_dataset, args):
    model.train()
    best_dcg, epoch_count = -1e6, 0
    step = 0
    best_model = None
    dcg_sum = 0.0
    for epoch_id in range(epoch_num):
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        for batch_id, batch in enumerate(train_dataloader):
            step += 1
            x, y = batch
            optimizer.zero_grad()
            dcg = model.fit(x, y, rel_num=1)
            optimizer.step()
            dcg_sum += dcg.item()
            if step % 1000 == 0:
                print(dcg_sum / 1000)
                dcg_sum = 0.0
            if step % 1000 == 0:
                valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=args.batch_size)
                with torch.no_grad():
                    valid_dcg = validate(model, valid_dataloader, metric=metric_name)
                    if best_model is None or valid_dcg > best_dcg:
                        best_model = copy.deepcopy(model)
                        best_dcg = valid_dcg
                        print("Best {} {} found in step {}".format(metric_name, best_dcg, step))
                        epoch_count = 0
                    else:
                        epoch_count += 1
                    if epoch_count >= 10:
                        print("Stop training at step {}".format(step))
                        return best_model
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=58)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--valid", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--comment", type=str, required=True)
    parser.add_argument("--lr", type=str, default="[1e-2, 1e-3, 1e-4]")
    parser.add_argument("--weight_decay", type=str, default="[1e-3, 1e-4, 1e-5, 1e-6, 1e-7]")
    parser.add_argument("--hidden_layers", type=int, default=1)
    parser.add_argument("--regression", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    mlp_config = {
        "input_size": args.input_dim,
        "hidden_layers": [],
        "final_size": 1
    }
    if args.regression:
        mlp_config['final_activation'] = "relu"
    last_size = args.input_dim
    for i in range(args.hidden_layers):
        mlp_config['hidden_layers'].append((last_size // 2, False, 0.0))
        last_size = last_size // 2
    normalizer = None
    if args.model is not None:
        model = LambdaRank(mlp_config)
        state_dict = torch.load(args.model)
        model.load_state_dict(state_dict)
        if args.test is not None:
            test_data = unserialize(os.path.join(args.test))
            test_dataset = RankingDataset(test_data, normalizer=normalizer, ranking=True)
            print(test_data[0][0][0])
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
            test_dcg = validate(model, test_dataloader,
                                output_path=os.path.join(args.output_dir, "prediction-{}.txt".format(args.comment)))
            print("DCG: {:.3f}".format(test_dcg))
            save_svmprop_train(test_dataset, path=os.path.join(args.output_dir, "test-{}.txt".format(args.comment)))
    if args.train is not None:
        # load the train data
        train_data = unserialize(args.train)
        # normalizer = DatasetNormalizer(centering=True)
        # normalizer.fit(train_data)
        normalizer = None
        print(train_data[0][0][0])
        global metric_name
        if args.regression:
            metric_name = "square"
            train_dataset = RankingDataset(train_data, normalizer=normalizer, divide_query=False, ranking=False)
        else:
            metric_name = "DCG"
            train_dataset = RankingDataset(train_data, normalizer=normalizer, divide_query=True, ranking=True)
        # load the valid data
        valid_data = unserialize(args.valid)
        print(valid_data[0][0][0])
        valid_dataset = RankingDataset(valid_data, normalizer=normalizer, divide_query=False, ranking=False)
        # load the test data
        if args.test is not None:
            test_data = unserialize(os.path.join(args.test))
            test_dataset = RankingDataset(test_data, normalizer=normalizer, divide_query=False, ranking=False)
            print(test_data[0][0][0])
        # Train
        epoch_num = 100
        print("Epoch num: {}".format(epoch_num))
        lrs, weight_decays = eval(args.lr), eval(args.weight_decay)
        best_dcg, best_parameters = 0.0, None
        for lr in lrs:
            for weight_decay in weight_decays:
                if args.regression:
                    model = Regression(mlp_config)
                else:
                    model = LambdaRank(mlp_config)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
                model = train(model, optimizer, train_dataset, epoch_num, valid_dataset, args)
                valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=args.batch_size)
                valid_dcg = validate(model, valid_dataloader, metric=metric_name)
                print("Valid {}: {:3f}".format(metric_name, valid_dcg))
                if best_parameters is None or valid_dcg > best_dcg:
                    best_dcg = valid_dcg
                    best_parameters = (lr, weight_decay)
                    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
                    test_dcg = validate(model, test_dataloader,
                                        output_path=os.path.join(args.output_dir,
                                                                 "prediction-{}.txt".format(args.comment)),
                                        metric=metric_name)
                    print("Lr: {}, Weight decay: {}, Test {}: {:.3f}".format(lr, weight_decay, metric_name, test_dcg))
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "state_dict-{}".format(args.comment)))
        print("The best hyperparameters are lr {} and weight decay {}, best test {} {} ".format(best_parameters[0],
                                                                                                best_parameters[1],
                                                                                                metric_name, test_dcg))


if __name__ == "__main__":
    main()
