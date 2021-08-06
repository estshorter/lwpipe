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
        logger.info(f"Loading from {filepath}")
        return np.load(filepath)

    def dump_npy(data, filepath: str | PurePath):
        _make_dir(Path(filepath).parent)
        logger.info(f"Dumping to {filepath}")
        np.save(filepath, data)

    def load_savez_compressed(filepath: str | PurePath, datalabels):
        logger.info(f"Loading from {filepath}")
        npz = np.load(filepath)
        _assert_same_length(datalabels, npz, "datalabels", "npz")
        datalist = [npz[label] for label in datalabels]
        return datalist

    def dump_savez_compressed(datalist, filepath: str | PurePath, datalabels):
        logger.info(f"Dumping to {filepath}")
        _assert_same_length(datalabels, datalist, "datalabels", "datalist")
        _make_dir(Path(filepath).parent)
        datadict = {label: data for label, data in zip(datalabels, datalist)}
        np.savez_compressed(filepath, **datadict)


def load_pickle(filepath: str | PurePath):
    logger.info(f"Loading from {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


def dump_pickle(data, filepath: str | PurePath):
    logger.info(f"Dumping to {filepath}")
    _make_dir(Path(filepath).parent)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_dict_pickle(filepath: str | PurePath, datalabels):
    logger.info(f"Loading from {filepath}")
    with open(filepath, "rb") as f:
        datadict = pickle.load(f)
        _assert_same_length(datalabels, datadict, "datalabels", "datadict")
        datalist = [datadict[label] for label in datalabels]
        return datalist


def dump_dict_pickle(datalist, filepath: str | PurePath, datalabels):
    logger.info(f"Dumping to {filepath}")
    _assert_same_length(datalabels, datalist, "datalabels", "datalist")
    datadict = {label: data for label, data in zip(datalabels, datalist)}
    _make_dir(Path(filepath).parent)
    with open(filepath, "wb") as f:
        pickle.dump(datadict, f)


def _make_dir(path: PurePath):
    if not path.exists():
        path.mkdir(parents=True)
