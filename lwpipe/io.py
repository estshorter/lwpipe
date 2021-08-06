from __future__ import annotations

import importlib.util
import logging
import pickle
from pathlib import Path, PurePath

from .utils import _assert_same_length

logger = logging.getLogger(__name__)


if importlib.util.find_spec("numpy"):
    import numpy as np

    def load_npy(filepath: str | PurePath):
        data = np.load(filepath)
        logger.info(f"Loaded from '{filepath}'")
        return data

    def dump_npy(data, filepath: str | PurePath):
        _make_dir(Path(filepath).parent)
        np.save(filepath, data)
        logger.info(f"Dumped to '{filepath}'")

    def load_savez_compressed(filepath: str | PurePath, datalabels):
        npz = np.load(filepath)
        _assert_same_length(datalabels, npz, "datalabels", "npz")
        datalist = [npz[label] for label in datalabels]
        logger.info(f"Loaded from '{filepath}'")
        return datalist

    def dump_savez_compressed(datalist, filepath: str | PurePath, datalabels):
        _assert_same_length(datalabels, datalist, "datalabels", "datalist")
        _make_dir(Path(filepath).parent)
        datadict = {label: data for label, data in zip(datalabels, datalist)}
        np.savez_compressed(filepath, **datadict)
        logger.info(f"Dumped to '{filepath}'")


def load_pickle(filepath: str | PurePath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded from '{filepath}'")
    return data


def dump_pickle(data, filepath: str | PurePath):
    _make_dir(Path(filepath).parent)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Dumped to '{filepath}'")


def load_dict_pickle(filepath: str | PurePath, datalabels):
    with open(filepath, "rb") as f:
        datadict = pickle.load(f)
    _assert_same_length(datalabels, datadict, "datalabels", "datadict")
    datalist = [datadict[label] for label in datalabels]
    logger.info(f"Loaded from '{filepath}'")
    return datalist


def dump_dict_pickle(datalist, filepath: str | PurePath, datalabels):
    _assert_same_length(datalabels, datalist, "datalabels", "datalist")
    datadict = {label: data for label, data in zip(datalabels, datalist)}
    _make_dir(Path(filepath).parent)
    with open(filepath, "wb") as f:
        pickle.dump(datadict, f)
    logger.info(f"Dumped to '{filepath}'")


def _make_dir(path: PurePath):
    if not path.exists():
        path.mkdir(parents=True)
