import numpy as np
from torch.autograd import Variable
import torch
import argparse
import json
import os

try:
    import cPickle as myPickle
except ImportError:
    import pickle as myPickle


def serialize(obj, path, in_json=False):
    if isinstance(obj, np.ndarray):
        np.save(path, obj)
    elif in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            myPickle.dump(obj, file)


def unserialize(path, form=None):
    if form is None:
        form = os.path.basename(path).split(".")[-1]
    if form == "npy":
        return np.load(path)
    elif form == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return myPickle.load(file)


def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def read_rank_dataset(path):
    with open(path) as file:
        for line in file:
            label, line = line.strip().split(' ', maxsplit=1)
            label = float(label)
            line = dict(list(map(lambda x: x.split(':'), line.split())))
            qid = int(line['qid'])
            features = {int(idx): float(value) for idx, value in line.items() if idx.isdigit()}
            cost = 1.0
            if 'cost' in line:
                cost = float(line['cost'])
            yield label, qid, features, cost


def transform_dataset(data, use_gpu, weighted):
    feats, rels = data
    feats, rels = torch.as_tensor(feats, dtype=torch.float), torch.as_tensor(rels, dtype=torch.float)
    if use_gpu:
        feats, rels = feats.cuda(), rels.cuda()
    if not weighted:
        rels = (rels > 0.0).float()
    return feats, rels


# from loss_utils import get_z_ids
def shuffle_combined(a, b):
    c = torch.randperm(len(a)).tolist()
    if torch.is_tensor(a):
        a = a[c]
    else:
        a = [a[i] for i in c]
    if torch.is_tensor(b):
        b = b[c]
    else:
        b = [b[i] for i in c]
    return a, b


def compute_test_errors(test_z, predictions):
    errors = [[0, 0], [0, 0]]
    total_errors = 0
    for i in [0, 1]:
        for j in [0, 1]:
            for i1 in test_z[i][j]:
                if not predictions[i1]:
                    errors[i][j] += 1
                    total_errors += 1
    for i in [0, 1]:
        for j in [0, 1]:
            errors[i][j] = float(errors[i][j]) / len(test_z[i][j])
    total_errors = float(total_errors) / len(predictions)
    return (total_errors, errors)


def get_error_rates(test_x, test_z_assignment, model):
    predictions = model(Variable(torch.FloatTensor(test_x[:, 0, :]))) - model(
        Variable(torch.FloatTensor(test_x[:, 1, :])))
    total_error = (torch.mean((predictions < 0).float())).data[0]
    # z_ass_1, z_ass_2, _, _, _, _ = get_z_ids(constraint_type)
    z0 = np.argwhere(np.asarray(test_z_assignment == 0)).flatten()
    z1 = np.argwhere(np.asarray(test_z_assignment == 1)).flatten()
    z2 = np.argwhere(np.asarray(test_z_assignment == 2)).flatten()
    z3 = np.argwhere(np.asarray(test_z_assignment == 3)).flatten()
    error0 = (torch.mean(
        (predictions < 0)[torch.LongTensor(z0)].float())).data[0]
    error1 = (torch.mean(
        (predictions < 0)[torch.LongTensor(z1)].float())).data[0]
    error2 = (torch.mean(
        (predictions < 0)[torch.LongTensor(z2)].float())).data[0]
    error3 = (torch.mean(
        (predictions < 0)[torch.LongTensor(z3)].float())).data[0]
    return total_error, error0, error1, error2, error3


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def postprocess_args(args):
    args.lr = [float(s) for s in args.lr.split(',')]
    args.baseline = args.baseline.split(',')
    args.sample_size = [int(s) for s in args.sample_size.split(',')]
    args.epochs = [int(s) for s in args.epochs.split(',')]

    args.weight_decay = [float(s) for s in args.weight_decay.split(',')]
    lens = [
        len(args.lr),
        len(args.sample_size),
        len(args.epochs),
        len(args.weight_decay),
        len(args.baseline)
    ]
    if np.any(np.array(lens) - max(lens) != 0):
        print("The number of args for Learning rate, sample size, epochs, "
              "baseline type and weight decay "
              "should be the same")
        import sys
        sys.exit(1)
    if args.eval_rank_limit < 1000:
        args.evalk = args.eval_rank_limit
        print("Override evalk with eval_rank_limit")
    return args


def torchify(u):
    return Variable(torch.FloatTensor(u))


def get_optimizer(params, lr, name="Adam", weight_decay=False, momentum=0.9):
    name = name.lower()
    from torch import optim
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adadelta":
        return optim.Adadelta(params, lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(params, lr, momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError


def exp_lr_scheduler(optimizer,
                     epoch,
                     init_lr=0.001,
                     decay_factor=0.1,
                     lr_decay_epoch=6,
                     min_lr=1e-5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (decay_factor ** (epoch // lr_decay_epoch))
    if lr < min_lr:
        lr = min_lr

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

# """
# The code below makes the print command work well with tqdm
# """
# import inspect
# from tqdm import tqdm
#
# # store builtin print
# old_print = print
#
#
# def new_print(*args, **kwargs):
#     # if tqdm.tqdm.write raises error, use builtin print
#     try:
#         tqdm.write(*args, **kwargs)
#     except:
#         old_print(*args, **kwargs)
#
#
# # globaly replace print with new_print
# inspect.builtins.print = new_print
# """
# """
