"""Data loading for the CATCH / TAB multivariate anomaly-detection benchmark files.

Each dataset is one long-format CSV (columns ``date, data, cols``) in which every channel is stored as one contiguous
block of T values and the last block is the point-wise label.  The train / test boundary is ``train_lens`` in
``DETECT_META.csv``.  We use the same files and boundaries as CATCH (ICLR 2025) and TAB (PVLDB 2025).

Regions of one series:

    [0, fit_end)           training windows; the scaler is fitted on this region only
    [fit_end, train_end)   validation windows (last ``validation_ratio`` of the training part), early stopping
    [train_end, T)         test, scored with non-overlapping windows; an incomplete tail (< seq_len) is not scored

    train_end = train_lens,   fit_end = round((1 - validation_ratio) * train_end)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Series:
    name: str
    values: np.ndarray      # (T, C) float32
    labels: np.ndarray      # (T,) int64, only read for evaluation
    train_end: int

    @property
    def channels(self) -> int:
        return self.values.shape[1]


def _read_long_csv(path: Path):
    points, first = 0, None                                         # length of the first channel block
    for chunk in pd.read_csv(path, usecols=["cols"], chunksize=1_000_000):
        names = chunk["cols"].to_numpy()
        first = names[0] if first is None else first
        change = np.flatnonzero(names != first)
        if len(change):
            points += int(change[0])
            break
        points += len(names)
    data = pd.read_csv(path, usecols=["data"], dtype={"data": np.float32})["data"].to_numpy()
    if len(data) % points:
        raise ValueError(f"{path}: {len(data)} values are not a multiple of the block length {points}")
    matrix = data.reshape(len(data) // points, points).T            # (T, C + 1), last column = label
    return np.ascontiguousarray(matrix[:, :-1], dtype=np.float32), np.asarray(matrix[:, -1], dtype=np.int64)


def load_series(root_path: str, dataset: str) -> Series:
    """Read ``{root}/data/{dataset}.csv`` once and cache it as .npy under ``{root}/cache``."""
    root = Path(root_path)
    meta = pd.read_csv(root / "DETECT_META.csv")
    row = meta.loc[meta["file_name"] == f"{dataset}.csv"]
    if len(row) != 1:
        raise ValueError(f"{dataset}.csv is not listed in {root / 'DETECT_META.csv'}")
    cache = root / "cache"
    values_path, labels_path = cache / f"{dataset}_values.npy", cache / f"{dataset}_labels.npy"
    if not (values_path.exists() and labels_path.exists()):
        cache.mkdir(parents=True, exist_ok=True)
        values, labels = _read_long_csv(root / "data" / f"{dataset}.csv")
        # Write under temporary names and rename, so an interrupted run never leaves a truncated cache behind.
        for path, array in ((values_path, values), (labels_path, labels)):
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
            np.save(tmp, array)
            tmp.replace(path)
    values = np.load(values_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    return Series(dataset, values, labels, int(row.iloc[0]["train_lens"]))


def fit_scaler(values: np.ndarray, fit_end: int):
    """Per-channel mean / std on [0, fit_end); a channel with zero std is divided by 1."""
    fit = np.asarray(values[:fit_end])
    mean = fit.mean(axis=0, dtype=np.float64)
    std = fit.std(axis=0, dtype=np.float64)
    std = np.where(std == 0.0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class WindowDataset(Dataset):
    """Standardised windows values[s : s + seq_len] for s in range(start_min, start_max + 1, stride).

    Items are (window (seq_len, C) float32, global start index); the start index lets scores be written back to the
    time axis.
    """

    def __init__(self, values, seq_len, start_min, start_max, mean, std, stride):
        if start_max < start_min:
            raise ValueError(f"Region too short for seq_len={seq_len}")
        self.values, self.seq_len = values, seq_len
        self.starts = np.arange(start_min, start_max + 1, stride, dtype=np.int64)
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        window = (np.asarray(self.values[start:start + self.seq_len], dtype=np.float32) - self.mean) / self.std
        return torch.from_numpy(window.copy()), torch.tensor(start, dtype=torch.long)


def build_datasets(series: Series, seq_len: int, validation_ratio: float, train_stride: int):
    """Return train / validation / test window sets, the scaler, and the scored test interval [train_end, test_end)."""
    train_end = series.train_end
    fit_end = int(round(train_end * (1.0 - validation_ratio)))
    if fit_end < seq_len or train_end - fit_end < seq_len or len(series.values) - train_end < seq_len:
        raise ValueError("Split too short for the requested seq_len")
    mean, std = fit_scaler(series.values, fit_end)
    train = WindowDataset(series.values, seq_len, 0, fit_end - seq_len, mean, std, train_stride)
    valid = WindowDataset(series.values, seq_len, fit_end, train_end - seq_len, mean, std, 1)
    count = (len(series.values) - train_end) // seq_len                   # non-overlapping test windows
    test = WindowDataset(series.values, seq_len, train_end, train_end + (count - 1) * seq_len, mean, std, seq_len)
    return train, valid, test, (train_end, train_end + count * seq_len)
