from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path, PurePath

if importlib.util.find_spec("numpy"):
    import numpy as np

    def load_npy(filepath: str | PurePath):
        return np.load(filepath)

    def dump_npy(data, filepath: str | PurePath):
        _make_dir(Path(filepath).parent)
        np.save(filepath, data)

    def load_savez_compressed(filepath: str | PurePath, names):
        npz = np.load(filepath)
        datalist = []
        for name in names:
            datalist.append(npz[name])
        return datalist

    def dump_savez_compressed(datalist, filepath: str | PurePath, names):
        _make_dir(Path(filepath).parent)
        datadict = {}
        for name, data in zip(names, datalist):
            datadict[name] = data
        np.savez_compressed(filepath, **datadict)


def load_pickle(filepath: str | PurePath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def dump_pickle(data, filepath: str | PurePath):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_dict_pickle(filepath: str | PurePath, names):
    datalist = []
    with open(filepath, "rb") as f:
        datadict = pickle.load(f)
        for name in names:
            datalist.append(datadict[name])
    return datalist


def dump_dict_pickle(datalist, filepath: str | PurePath, names):
    datadict = {}
    for name, data in zip(names, datalist):
        datadict[name] = data

    with open(filepath, "wb") as f:
        pickle.dump(datadict, f)


def _make_dir(path: PurePath):
    if not path.exists():
        path.mkdir(parents=True)
