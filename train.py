"""Обучает SVM и CNN, выбирает настройку CNN на валидации и оценивает тест."""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import dct
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from threadpoolctl import threadpool_limits

from audio_features import GENRES, ROOT
from model import GenreCNN, normalized_tensor

EXPERIMENTS = (
    ("cnn_clean", False, 0.001),
    ("cnn_aug", True, 0.001),
    ("cnn_aug_low_lr", True, 0.0004),
)
BATCH = 64
SEED = 42


def aggregate(scores, data):
    ids, starts, lengths = np.unique(data["track"], return_index=True, return_counts=True)
    predictions = np.add.reduceat(scores, starts, axis=0) / lengths[:, None]
    return ids, data["y"][starts], predictions


def metrics(y, predictions):
    report = classification_report(
        y, predictions, labels=np.arange(len(GENRES)), target_names=GENRES,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro": report["macro avg"], "classes": {key: report[key] for key in GENRES},
        "confusion": confusion_matrix(y, predictions, labels=np.arange(len(GENRES))).tolist(),
    }


def evaluate_scores(scores, data):
    _, targets, track_scores = aggregate(scores, data)
    return {
        "clip": metrics(data["y"], scores.argmax(axis=1)),
        "track": metrics(targets, track_scores.argmax(axis=1)),
    }


def mfcc_stats(features):
    coefficients = dct(features.astype("float32"), type=2, axis=1, norm="ortho")[:, :20]
    deltas = np.diff(coefficients, axis=2)
    return np.concatenate((
        coefficients.mean(axis=2), coefficients.std(axis=2),
        deltas.mean(axis=2), deltas.std(axis=2),
    ), axis=1)


def get_split(output, split):
    with np.load(output / f"{split}.npz") as file:
        return {key: file[key] for key in file.files}


@torch.inference_mode()
def evaluate_cnn(model, values, targets, loss_fn):
    model.eval()
    loss = 0.0
    scores = []
    for start in range(0, len(targets), BATCH):
        end = start + BATCH
        logits = model(values[start:end])
        labels = targets[start:end]
        loss += loss_fn(logits, labels).item() * len(labels)
        scores.append(logits.softmax(dim=1).numpy())
    return np.concatenate(scores), loss / len(targets)


def fit(name, x, y, x_aug, x_valid, valid, lr, epochs, output):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    augmentation_rng = np.random.default_rng(SEED + 1)
    model = GenreCNN().to(memory_format=torch.channels_last)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0003)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)
    y_valid = torch.as_tensor(valid["y"], dtype=torch.long)
    best_key = (-1.0, float("-inf"))
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        order = rng.permutation(len(y))
        for start in range(0, len(y), BATCH):
            indices = order[start : start + BATCH]
            inputs = x[indices]
            if x_aug is not None:
                use_aug = augmentation_rng.random(len(indices)) < 0.5
                inputs[use_aug] = x_aug[indices[use_aug]]
            inputs = inputs.contiguous(memory_format=torch.channels_last)
            labels = y[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        scores, val_loss = evaluate_cnn(model, x_valid, y_valid, loss_fn)
        validation = evaluate_scores(scores, valid)
        val_f1 = validation["track"]["macro"]["f1-score"]
        history.append({
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": running_loss / len(y), "train_accuracy": correct / len(y),
            "valid_loss": val_loss, "valid_accuracy": validation["clip"]["accuracy"],
            "valid_track_f1": val_f1, "seconds": time.perf_counter() - started,
        })
        print(f"{name} {epoch:02d}/{epochs}  loss {running_loss / len(y):.3f}"
              f"  val F1 {val_f1:.3f}  ({history[-1]['seconds']:.1f}s)", flush=True)
        schedule.step()
        key = (val_f1, -val_loss)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_validation = validation
            torch.save(model.state_dict(), output / f"{name}.pt")
        if epoch - best_epoch >= 7:
            break
    return {"best_epoch": best_epoch, "validation": best_validation, "history": history}


def save_predictions(path, scores, data, unit):
    keys, labels, values = aggregate(scores, data) if unit == "track" else (
        data["track"], data["y"], scores,
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["track", "start_seconds", "expected", "predicted", *GENRES])
        last = None
        for key, label, row in zip(keys, labels, values):
            start = start + 6 if key == last else 0
            writer.writerow([key, start if unit == "clip" else "", GENRES[label],
                             GENRES[row.argmax()], *[f"{score:.6f}" for score in row]])
            last = key


def plots(output, history, confusion):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    epochs = [item["epoch"] for item in history]
    for ax, key, ylabel in zip(axes, ["loss", "accuracy"], ["Кросс-энтропия", "Accuracy"]):
        for split, label, color in [
            ("train", "Обучение", "#087F8C"), ("valid", "Валидация", "#C7572B"),
        ]:
            ax.plot(epochs, [item[f"{split}_{key}"] for item in history],
                    label=label, color=color, linewidth=2)
        ax.set(xlabel="Эпоха", ylabel=ylabel)
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.savefig(output / "learning.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 5.8), layout="constrained")
    matrix = np.asarray(confusion)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(GENRES)), xticklabels=GENRES, yticks=range(len(GENRES)),
           yticklabels=GENRES, xlabel="Предсказанный жанр", ylabel="Истинный жанр")
    ax.tick_params(axis="x", labelrotation=36)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center",
                color="white" if value > matrix.max() * 0.5 else "#263648", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.87, label="Число записей")
    fig.savefig(output / "confusion.png", dpi=170)
    plt.close(fig)


def train(output, epochs, threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    train_data, valid = (get_split(output, split) for split in ("train", "valid"))
    mean = train_data["x"].mean(axis=(0, 2), dtype=np.float64).astype("float32")[:, None]
    scale = train_data["x"].std(axis=(0, 2), dtype=np.float64).astype("float32")[:, None]
    scale = np.maximum(scale, 1e-5)
    x = normalized_tensor(train_data["x"], mean, scale)
    x_aug = normalized_tensor(train_data["augmented"], mean, scale)
    y = torch.as_tensor(train_data["y"], dtype=torch.long)
    x_valid = normalized_tensor(valid["x"], mean, scale)
    with threadpool_limits(limits=1, user_api="blas"):
        baseline = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
        baseline.fit(mfcc_stats(train_data["x"]), train_data["y"])
        baseline_val = baseline.decision_function(mfcc_stats(valid["x"]))
    del train_data
    results = {
        "seed": SEED, "batch": BATCH, "max_epochs": epochs,
        "parameters": sum(param.numel() for param in GenreCNN().parameters()),
        "experiments": {"svm_mfcc": {"validation": evaluate_scores(baseline_val, valid)}},
    }
    experiments = results["experiments"]
    for name, use_aug, lr in EXPERIMENTS:
        experiments[name] = fit(name, x, y, x_aug if use_aug else None, x_valid,
                                valid, lr, epochs, output)
        experiments[name]["initial_lr"] = lr
    selected = max(
        (name for name, _, _ in EXPERIMENTS),
        key=lambda name: experiments[name]["validation"]["track"]["macro"]["f1-score"],
    )
    results["selected"] = selected
    test = get_split(output, "test")
    experiments["svm_mfcc"]["test"] = evaluate_scores(
        baseline.decision_function(mfcc_stats(test["x"])), test,
    )
    x_test = normalized_tensor(test["x"], mean, scale)
    y_test = torch.as_tensor(test["y"], dtype=torch.long)
    for name, _, _ in EXPERIMENTS:
        model = GenreCNN().to(memory_format=torch.channels_last).eval()
        state = torch.load(output / f"{name}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        scores, _ = evaluate_cnn(model, x_test, y_test, nn.CrossEntropyLoss())
        experiments[name]["test"] = evaluate_scores(scores, test)
        if name == selected:
            torch.save({
                "version": 1, "state": state, "genres": list(GENRES),
                "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
            }, output / "genre_cnn.pt")
            for unit in ("clip", "track"):
                save_predictions(output / f"test_{unit}.csv", scores, test, unit)
    chosen = experiments[selected]
    plots(output, chosen["history"], chosen["test"]["track"]["confusion"])
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Selected {selected}, test track F1 {chosen['test']['track']['macro']['f1-score']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs и threads должны быть положительными")
    train(args.output, args.epochs, args.threads)
