import numpy as np
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