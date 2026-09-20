"""Train and export OmniGuard's lightweight AI-text classifier.

The source dataset is ``rasbt/human-vs-ai-50k`` (Apache-2.0).  This script
intentionally trains from Parquet rather than loading the published joblib
artifact: pickle/joblib files can execute code while loading.  The exported
runtime format contains only JSON metadata and NumPy numeric arrays.

Training-only dependencies::

    pip install scikit-learn pyarrow

Example::

    python tools/train_text_detector.py \
      --train train.parquet --validation validation.parquet \
      --test test.parquet --output backend/text_models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def _read(path: Path) -> tuple[list[str], np.ndarray]:
    table = pq.read_table(path, columns=["text", "label"])
    data = table.to_pydict()
    return data["text"], np.asarray(data["label"], dtype=np.int64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    positive = x >= 0
    result = np.empty_like(x, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_text, train_y = _read(args.train)
    validation_text, validation_y = _read(args.validation)
    test_text, test_y = _read(args.test)

    # Bound the vocabulary so the deployed dict stays comfortably below the
    # Render free tier's 512 MB RAM limit.  The most frequent 200k unigrams and
    # bigrams retain nearly all test accuracy while using a fraction of the
    # runtime memory of the unrestricted 1.3M-feature vocabulary.
    vectorizer = TfidfVectorizer(
        min_df=2, ngram_range=(1, 2), max_features=200_000
    )
    train_x = vectorizer.fit_transform(train_text)
    validation_x = vectorizer.transform(validation_text)
    test_x = vectorizer.transform(test_text)

    classifier = LogisticRegression(C=3.0, max_iter=1000, random_state=42)
    classifier.fit(train_x, train_y)

    # Platt calibration: learn a one-dimensional sigmoid from the untouched
    # validation split.  Keeping it explicit makes the runtime implementation
    # dependency-free and auditable.
    validation_raw = classifier.decision_function(validation_x).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=42)
    calibrator.fit(validation_raw, validation_y)

    test_raw = classifier.decision_function(test_x)
    calibrated_logit = (
        calibrator.coef_[0, 0] * test_raw + calibrator.intercept_[0]
    )
    test_probability = _sigmoid(calibrated_logit)
    test_prediction = (test_probability >= 0.5).astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(test_y, test_prediction)),
        "roc_auc": float(roc_auc_score(test_y, test_probability)),
        "confusion_matrix": confusion_matrix(test_y, test_prediction).tolist(),
        "train_samples": len(train_text),
        "validation_samples": len(validation_text),
        "test_samples": len(test_text),
        "features": len(vectorizer.vocabulary_),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    vocabulary = [None] * len(vectorizer.vocabulary_)
    for term, index in vectorizer.vocabulary_.items():
        vocabulary[index] = term

    (args.output / "vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output / "classifier.npz",
        idf=vectorizer.idf_.astype(np.float32),
        coefficients=classifier.coef_[0].astype(np.float32),
        intercept=np.asarray(classifier.intercept_, dtype=np.float32),
        calibration_coefficient=np.asarray(
            [calibrator.coef_[0, 0]], dtype=np.float32
        ),
        calibration_intercept=np.asarray(
            calibrator.intercept_, dtype=np.float32
        ),
    )
    metadata = {
        "format_version": 1,
        "model": "TF-IDF (top 200k word 1-2 grams) + logistic regression + Platt scaling",
        "dataset": "rasbt/human-vs-ai-50k",
        "dataset_license": "Apache-2.0",
        "labels": {"0": "human", "1": "ai"},
        "decision_threshold": 0.5,
        "minimum_words": 40,
        "metrics": metrics,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
