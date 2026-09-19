"""Small, dependency-free checks for the Colab training data contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


TRAINING = Path(__file__).resolve().parents[2] / "training" / "train.py"
SPEC = importlib.util.spec_from_file_location("omniguard_training", TRAINING)
training = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(training)


class _Split:
    column_names = ["image", "label", "pair_id"]

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def select(self, indexes):
        return _Split([self.rows[index] for index in indexes])


def test_paired_samples_never_cross_data_splits():
    rows = [
        {"image": f"{kind}_{pair}.jpg", "label": int(kind == "fake"),
         "pair_id": str(pair)}
        for pair in range(12)
        for kind in ("real", "fake")
    ]
    args = SimpleNamespace(n_train=12, n_val=6, n_test=6, seed=42)

    splits = training.make_splits({"train": _Split(rows)}, args)
    ids = {name: set(split["pair_id"]) for name, split in splits.items()}

    assert len(splits["train"]) == 12
    assert len(splits["val"]) == 6
    assert len(splits["test"]) == 6
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])


def test_default_datasets_match_each_training_task():
    assert training.DATASET_CANDIDATES["face_manipulation"] == ["Sowaiba01/Deepfake"]
    assert training.DATASET_CANDIDATES["ai_generation"] == [
        "zr-zhang/MLLM-Generated-Image-Detection-Dataset"
    ]
    assert training.DATASET_LICENSES["Sowaiba01/Deepfake"] == "MIT"
    assert (
        training.DATASET_LICENSES["zr-zhang/MLLM-Generated-Image-Detection-Dataset"]
        == "Apache-2.0"
    )
