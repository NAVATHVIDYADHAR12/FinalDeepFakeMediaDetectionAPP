"""OmniGuard AI - deepfake detector training.

This is the single source of truth for training. The Colab notebook writes this
exact file out and runs it, so editing here changes what Colab runs.

Typical use (Colab, free T4 GPU):
    python train.py

Faster smoke test:
    python train.py --n-train 2000 --n-val 500 --epochs 1 --models efficientnet_b0

Everything is overridable:
    python train.py --epochs 5 --batch-size 32 --img-size 256

Outputs, all written to --out (default ./out):
    models/<arch>.onnx                 classifier, 2 outputs: logits + feature map
    models/<arch>_classifier_w.npy     head weights, needed for CAM heatmaps
    models/manifest.json               config + test-set metrics
    metrics/*.png                      ROC, confusion matrix, model comparison
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Class index convention shared with the backend: 0 = REAL, 1 = FAKE
REAL_IDX, FAKE_IDX = 0, 1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Keep the two visual tasks explicit. A face-swap corpus cannot validate an
# AI-image detector, and an AI-art corpus cannot validate face manipulation.
DATASET_CANDIDATES = {
    "face_manipulation": ["Sowaiba01/Deepfake"],
    "ai_generation": ["zr-zhang/MLLM-Generated-Image-Detection-Dataset"],
}
DATASET_LICENSES = {
    "Sowaiba01/Deepfake": "MIT",
    "julienlucas/midjourney-dalle-sd-dataset": "MIT",
    "zr-zhang/MLLM-Generated-Image-Detection-Dataset": "Apache-2.0",
}

DEFAULT_MODELS = ["efficientnet_b0", "legacy_xception", "mobilenetv3_large_100"]


# --------------------------------------------------------------------- arguments
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train deepfake detection models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", choices=sorted(DATASET_CANDIDATES),
                   default="face_manipulation",
                   help="train one explicit detection task; never mix their scores")
    p.add_argument("--n-train", type=int, default=8000,
                   help="training images to use (the full set is 10,852)")
    p.add_argument("--n-val", type=int, default=1400)
    p.add_argument("--n-test", type=int, default=1400)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64,
                   help="lower to 32 if the GPU runs out of memory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--label-smooth", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="timm architecture names to train")
    p.add_argument("--dataset", default=None,
                   help="force a specific HuggingFace dataset id")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--zip", action="store_true",
                   help="also produce omniguard_models.zip")
    p.add_argument("--no-pretrained", action="store_true",
                   help="train from scratch (much worse; for ablation only)")
    p.add_argument("--allow-cpu", action="store_true",
                   help="permit CPU training (intended only for tiny smoke tests)")
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------- dataset
def load_source(forced: str | None, task: str):
    # Repositories with thousands of small image files can otherwise emit one
    # progress update per file.  That is harmless in a terminal but can make a
    # Colab notebook tab effectively unresponsive while the GPU runtime keeps
    # working in the background.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BARS", "1")

    from datasets import Dataset, DatasetDict, Image as HFImage, disable_progress_bars, load_dataset
    from huggingface_hub import snapshot_download

    disable_progress_bars()

    candidates = [forced] if forced else DATASET_CANDIDATES[task]
    for repo in candidates:
        try:
            print(f"  trying {repo} ...", flush=True)
            if repo == "Sowaiba01/Deepfake":
                # The repository is a plain image-folder dataset. Downloading
                # only its two class directories avoids a broken optional
                # metadata file in the repository root.
                local = snapshot_download(
                    repo_id=repo, repo_type="dataset",
                    allow_patterns=["fake/*.jpg", "real/*.jpg"],
                    max_workers=32,
                )
                # Hugging Face snapshots expose their objects as symlinks.
                # The imagefolder builder ignores those in some Colab/datasets
                # combinations, so build the small metadata table explicitly.
                fake_paths = sorted(Path(local).glob("fake/*.jpg"))
                real_paths = sorted(Path(local).glob("real/*.jpg"))
                image_paths = fake_paths + real_paths
                if not fake_paths or not real_paths:
                    raise RuntimeError("downloaded snapshot has no class images")
                train = Dataset.from_dict({
                    "image": [str(path) for path in image_paths],
                    "label": [0] * len(fake_paths) + [1] * len(real_paths),
                    # fake_N and real_N are paired. Keep N so make_splits can
                    # prevent the same identity leaking into train and test.
                    "pair_id": [path.stem.rsplit("_", 1)[-1]
                                for path in image_paths],
                }).cast_column("image", HFImage(decode=True))
                ds = DatasetDict({"train": train})
            else:
                ds = load_dataset(repo)
            print(f"  loaded {repo}")
            return ds, repo
        except Exception as exc:                           # noqa: BLE001
            print(f"    unavailable ({type(exc).__name__}). Next.")
    raise SystemExit(
        "No dataset source could be reached. Check the internet connection, "
        "or pass --dataset <huggingface-id>."
    )


def resolve_label_column(ds) -> tuple[str, list[str], set[int]]:
    """Find the label column and all stored classes that mean synthetic.

    Modern benchmarks often expose one class per generator (for example
    ``GPT-Image2-fake`` and ``Nano-Banana2-fake``), not a single fake class.
    They are collapsed into the backend's binary REAL/FAKE convention here.
    """
    split = ds["train"]
    label_col = next(
        (c for c in ("label", "labels", "class", "target") if c in split.column_names),
        None,
    )
    if label_col is None:
        raise SystemExit(f"No label column found in {split.column_names}")

    names = list(getattr(split.features[label_col], "names", ["Fake", "Real"]))
    fake_indices = {
        i for i, name in enumerate(names)
        if any(token in str(name).strip().lower()
               for token in ("fake", "synthetic", "ai-generated", "deepfake"))
    }
    if not fake_indices:
        raise SystemExit(
            f"Could not identify the fake class from labels {names}. "
            "Refusing to guess the class mapping."
        )
    return label_col, names, fake_indices


def make_splits(ds, args) -> dict:
    """Carve train/val/test, generating held-out splits if the source lacks them."""
    if "pair_id" in ds["train"].column_names:
        source = ds["train"]
        pair_ids = sorted(set(source["pair_id"]))
        random.Random(args.seed).shuffle(pair_ids)
        requested_pairs = {
            "train": args.n_train // 2,
            "val": args.n_val // 2,
            "test": args.n_test // 2,
        }
        if sum(requested_pairs.values()) > len(pair_ids):
            raise SystemExit(
                f"Requested {sum(requested_pairs.values()) * 2:,} paired images, "
                f"but the dataset contains only {len(pair_ids) * 2:,}."
            )
        groups, cursor = {}, 0
        for name, count in requested_pairs.items():
            groups[name] = set(pair_ids[cursor:cursor + count])
            cursor += count
        splits = {
            name: source.select([
                i for i, pair_id in enumerate(source["pair_id"])
                if pair_id in ids
            ])
            for name, ids in groups.items()
        }
    else:
        def take(name, n, skip=0):
            d = ds[name].shuffle(seed=args.seed)
            end = min(skip + n, len(d))
            return d.select(range(min(skip, len(d)), end))

        train = take("train", args.n_train)

        if "validation" in ds:
            val = take("validation", args.n_val)
        else:
            val = take("train", args.n_val, skip=args.n_train)

        if "test" in ds:
            test = take("test", args.n_test)
        else:
            test = take("train", args.n_test, skip=args.n_train + args.n_val)

        splits = {"train": train, "val": val, "test": test}
    for name, split in splits.items():
        if len(split) == 0:
            raise SystemExit(
                f"The requested {name} split is empty. Reduce --n-train, "
                "--n-val, or --n-test."
            )
    return splits


def build_loaders(splits, label_col, fake_indices, args):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
    from PIL import Image

    size = args.img_size

    # Re-compression augmentation is the highest-value augmentation here: without
    # it the network learns JPEG artifacts rather than forgery artifacts, and
    # collapses the moment an image is re-saved or passed through social media.
    try:
        from torchvision.transforms import v2
        jpeg = [v2.RandomApply([v2.JPEG(quality=(40, 95))], p=0.5)]
        print("  JPEG augmentation: on")
    except Exception:                                      # noqa: BLE001
        jpeg = []
        print("  JPEG augmentation: unavailable in this torchvision, skipped")

    train_tf = T.Compose([
        T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.25),
        *jpeg,
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    class DeepfakeDataset(Dataset):
        def __init__(self, hf_split, tf):
            self.d, self.tf = hf_split, tf

        def __len__(self):
            return len(self.d)

        def __getitem__(self, i):
            row = self.d[i]
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            y = FAKE_IDX if row[label_col] in fake_indices else REAL_IDX
            return self.tf(img.convert("RGB")), y

    def loader(split, tf, shuffle):
        return DataLoader(
            DeepfakeDataset(split, tf),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
            persistent_workers=args.workers > 0,
        )

    return (loader(splits["train"], train_tf, True),
            loader(splits["val"], eval_tf, False),
            loader(splits["test"], eval_tf, False))


# ------------------------------------------------------------------------- model
def build_model(arch: str, pretrained: bool):
    import timm
    import torch.nn as nn

    class CAMModel(nn.Module):
        """Backbone that also returns its final conv feature map.

        The backend needs that map to draw class-activation heatmaps. Exporting
        it as a second ONNX output means the explanation is computed from the
        same forward pass as the prediction, with no gradients required.
        """

        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=2)

        def forward(self, x):
            feats = self.backbone.forward_features(x)
            pooled = self.backbone.forward_head(feats, pre_logits=True)
            return self.backbone.get_classifier()(pooled), feats

        def classifier_weight(self):
            return self.backbone.get_classifier().weight.detach().cpu().numpy()

    return CAMModel()


# ---------------------------------------------------------------------- training
def evaluate(model, loader, device, amp):
    import torch
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=amp):
                logits, _ = model(x)
            probs.append(torch.softmax(logits.float(), 1)[:, FAKE_IDX].cpu())
            labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def train_one(arch, loaders, device, args):
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    train_dl, val_dl, _ = loaders
    amp = device == "cuda"

    print("\n" + "=" * 70)
    print(f"TRAINING  {arch}")
    print("=" * 70, flush=True)

    model = build_model(arch, pretrained=not args.no_pretrained).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = OneCycleLR(opt, max_lr=args.lr,
                       total_steps=max(1, args.epochs * len(train_dl)), pct_start=0.25)
    lossf = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_acc, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total_loss, seen, correct = time.time(), 0.0, 0, 0

        for step, (x, y) in enumerate(train_dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                logits, _ = model(x)
                loss = lossf(logits, y)
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # GradScaler skips optimizer.step when it detects an overflow.  In
            # that case advancing OneCycleLR would put the schedule one update
            # ahead of the weights and emits PyTorch's scheduler-order warning.
            if scaler.get_scale() >= scale_before:
                sched.step()

            total_loss += loss.item() * y.size(0)
            seen += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()

            if step % 100 == 0:
                frac = (step + 1) / len(train_dl)
                eta = (time.time() - t0) / max(frac, 1e-9) * (1 - frac)
                print(f"  epoch {epoch}  {frac * 100:5.1f}%  "
                      f"loss {total_loss / seen:.4f}  acc {correct / seen:.4f}  "
                      f"eta {eta / 60:.1f}m", flush=True)

        probs, labels = evaluate(model, val_dl, device, amp)
        val_acc = float(((probs > 0.5).astype(int) == labels).mean())
        print(f"  epoch {epoch} finished in {(time.time() - t0) / 60:.1f}m  "
              f"| val acc {val_acc:.4f}", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint_dir = args.out / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"arch": arch, "best_val_accuracy": best_acc, "state_dict": best_state},
                checkpoint_dir / f"{arch}.pt",
            )
            print(f"  best so far ({val_acc:.4f}) - checkpointed")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"DONE {arch}  best val accuracy {best_acc:.4f}")
    return model, best_acc


# ----------------------------------------------------------------------- metrics
def score(y_true, probs) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)
    pred = (probs > 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else 0.5,
    }


def write_charts(y_true, per_model, ensemble, results, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, confusion_matrix

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    # ROC
    plt.figure(figsize=(7, 6))
    for arch, probs in per_model.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        plt.plot(fpr, tpr, lw=1.8, label=f"{arch}  (AUC {results[arch]['roc_auc']:.3f})")
    fpr, tpr, _ = roc_curve(y_true, ensemble)
    plt.plot(fpr, tpr, lw=3, color="#22d3ee",
             label=f"ENSEMBLE  (AUC {results['ENSEMBLE']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC - Deepfake Detection")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, (ensemble > 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="magma")
    ax.set_xticks([0, 1], ["REAL", "FAKE"])
    ax.set_yticks([0, 1], ["REAL", "FAKE"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Ensemble")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Accuracy comparison
    names = list(results)
    accs = [results[n]["accuracy"] for n in names]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(names, accs, color=["#8b5cf6", "#ec4899", "#f59e0b", "#22d3ee"][:len(names)])
    plt.ylim(max(0.0, min(accs) - 0.05), 1.0)
    plt.ylabel("Test accuracy")
    plt.title("Model Comparison")
    plt.xticks(rotation=15, ha="right")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.003,
                 f"{acc:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close()

    print(f"  charts -> {out_dir}")


# ------------------------------------------------------------------------ export
def export_onnx(models, args, results, source, n_test, models_dir: Path) -> dict:
    import torch
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task": args.task,
        "source_dataset": source,
        "source_license": DATASET_LICENSES.get(source, "CHECK_DATASET_CARD"),
        "class_index": {"0": "REAL", "1": "FAKE"},
        "img_size": args.img_size,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        "metrics": results,
        "n_test_images": int(n_test),
        "models": [],
    }

    device = next(iter(models.values()))[0].parameters().__next__().device
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)

    for arch, (model, _) in models.items():
        model.eval()
        path = models_dir / f"{arch}.onnx"
        torch.onnx.export(
            model, dummy, str(path),
            input_names=["input"],
            output_names=["logits", "features"],
            dynamic_axes={"input": {0: "batch"},
                          "logits": {0: "batch"},
                          "features": {0: "batch"}},
            opset_version=18,
        )
        weights = model.classifier_weight()
        np.save(models_dir / f"{arch}_classifier_w.npy", weights)

        size_mb = path.stat().st_size / 1e6
        manifest["models"].append({
            "arch": arch,
            "file": f"{arch}.onnx",
            "classifier_weight_file": f"{arch}_classifier_w.npy",
            "classifier_weight_shape": list(weights.shape),
            "size_mb": round(size_mb, 1),
            "metrics": results.get(arch, {}),
        })
        print(f"  exported {arch:<24} {size_mb:6.1f} MB   head {weights.shape}")

    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_onnx(manifest, models, test_dl, models_dir: Path, device) -> bool:
    """Confirm the exported files reproduce PyTorch's predictions.

    A silent export mismatch would mean the laptop scores images differently
    from the reported test accuracy, so this is checked rather than assumed.
    """
    import torch
    import onnxruntime as ort

    xb, _ = next(iter(test_dl))
    sample = xb[:8]
    all_ok = True

    for entry in manifest["models"]:
        sess = ort.InferenceSession(str(models_dir / entry["file"]),
                                    providers=["CPUExecutionProvider"])
        logits, feats = sess.run(None, {"input": sample.numpy()})
        p_onnx = torch.softmax(torch.tensor(logits), 1)[:, FAKE_IDX].numpy()

        model = models[entry["arch"]][0].eval()
        with torch.no_grad():
            t_logits, _ = model(sample.to(device))
        p_torch = torch.softmax(t_logits.float().cpu(), 1)[:, FAKE_IDX].numpy()

        drift = float(np.abs(p_onnx - p_torch).max())
        ok = drift < 0.01
        all_ok &= ok
        print(f"  {entry['arch']:<24} features {feats.shape}  "
              f"max drift {drift:.5f}  {'OK' if ok else 'MISMATCH'}")

    return all_ok


# -------------------------------------------------------------------------- main
def main(argv=None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  OmniGuard AI - training")
    print("=" * 70)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("  WARNING: no GPU detected. On CPU this will be extremely slow.")
        print("  In Colab: Runtime -> Change runtime type -> T4 GPU.")
        if not args.allow_cpu:
            raise SystemExit(
                "No CUDA GPU detected. Refusing an accidental multi-day CPU run. "
                "Use a Colab T4 GPU, or pass --allow-cpu only for a tiny smoke test."
            )
    for k, v in vars(args).items():
        print(f"    {k:<16} {v}")

    print("\n[1/6] dataset")
    ds, source = load_source(args.dataset, args.task)
    label_col, class_names, fake_indices = resolve_label_column(ds)
    print(f"  label column '{label_col}', classes {class_names}, "
          f"stored fake indices {sorted(fake_indices)}")

    splits = make_splits(ds, args)
    for name, split in splits.items():
        counts = collections.Counter(split[label_col])
        pretty = {class_names[k] if k < len(class_names) else k: v
                  for k, v in sorted(counts.items())}
        print(f"  {name:<6} {len(split):>7,}  {pretty}")

    print("\n[2/6] data loaders")
    loaders = build_loaders(splits, label_col, fake_indices, args)
    print(f"  {len(loaders[0])} training batches per epoch")

    print("\n[3/6] training")
    trained = {}
    for arch in args.models:
        try:
            trained[arch] = train_one(arch, loaders, device, args)
        except Exception as exc:                           # noqa: BLE001
            print(f"\n  {arch} failed ({type(exc).__name__}: {exc}). "
                  f"Continuing with the rest.")
    if not trained:
        print("\nNo model trained successfully.")
        return 1

    print("\n[4/6] evaluation on the held-out test set")
    test_dl = loaders[2]
    per_model, results, y_true = {}, {}, None
    for arch, (model, _) in trained.items():
        probs, y_true = evaluate(model, test_dl, device, device == "cuda")
        per_model[arch] = probs
        results[arch] = score(y_true, probs)

    ensemble = np.mean(list(per_model.values()), axis=0)
    results["ENSEMBLE"] = score(y_true, ensemble)

    print(f"\n  {'MODEL':<26}{'ACC':>8}{'PREC':>8}{'REC':>8}{'F1':>8}{'AUC':>8}")
    print("  " + "-" * 66)
    for name, r in results.items():
        print(f"  {name:<26}{r['accuracy']:>8.4f}{r['precision']:>8.4f}"
              f"{r['recall']:>8.4f}{r['f1']:>8.4f}{r['roc_auc']:>8.4f}")

    write_charts(y_true, per_model, ensemble, results, args.out / "metrics")

    print("\n[5/6] ONNX export")
    models_dir = args.out / "models"
    manifest = export_onnx(trained, args, results, source, len(y_true), models_dir)

    print("\n[6/6] verifying the exported models")
    if not verify_onnx(manifest, trained, test_dl, models_dir, device):
        print("  WARNING: ONNX output drifted from PyTorch. Investigate before use.")

    shutil.copytree(args.out / "metrics", models_dir / "metrics", dirs_exist_ok=True)

    if args.zip:
        archive = shutil.make_archive("omniguard_models", "zip", str(models_dir))
        print(f"\n  packaged {archive} ({os.path.getsize(archive) / 1e6:.1f} MB)")

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    for name, r in results.items():
        print(f"    {name:<24} accuracy {r['accuracy']:.4f}   AUC {r['roc_auc']:.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
