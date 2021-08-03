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


if importlib.util.find_spec("pandas"):
    import pandas as pd

    def load_csv_as_dataframe(filepath: str | PurePath):
        return pd.read_csv(filepath)

    def load_dataframe(filepath: str | PurePath):
        return pd.read_pickle(filepath)

    def dump_dataframe(df: pd.DataFrame, filepath: str | PurePath):
        _make_dir(Path(filepath).parent)
        df.to_pickle(filepath)


def load_pickle(filepath: str | PurePath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def dump_pickle(data, filepath: str | PurePath):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def _make_dir(path: PurePath):
    if not path.exists():
        path.mkdir(parents=True)
