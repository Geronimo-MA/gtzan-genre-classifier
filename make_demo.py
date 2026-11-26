"""Выбирает и перепроверяет реальные тестовые записи для демонстрации."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app import analyze
from audio_features import GENRES, ROOT
from model import Predictor
from prepare import sources


def make_demo(dataset, output, artifacts):
    with (artifacts / "test_track.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    result = json.loads((artifacts / "results.json").read_text(encoding="utf-8"))
    matrix = np.array(result["experiments"][result["selected"]]["test"]["track"]["confusion"])
    np.fill_diagonal(matrix, 0)
    actual, predicted = np.unravel_index(matrix.argmax(), matrix.shape)
    pair = [row for row in rows
            if row["expected"] == GENRES[actual] and row["predicted"] == GENRES[predicted]]
    success = [row for row in rows if row["expected"] == row["predicted"]]
    mistakes = [row for row in rows if row["expected"] != row["predicted"]]

    def score(row):
        return float(row[row["predicted"]])

    selected = {}
    for group, choose in ((pair, max), (success, max), (mistakes, max), (success, min)):
        available = [row for row in group if row["track"] not in selected]
        if available:
            row = choose(available, key=score)
            selected[row["track"]] = row
    output.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    predictor = Predictor(artifacts / "genre_cnn.pt")
    verified = {}
    for key, source in sources(dataset, selected):
        wave, rate = sf.read(source, dtype="int16")
        path = output / f"{key}.wav"
        sf.write(path, wave, rate, subtype="PCM_16")
        case = {"file": path.name, "expected": selected[key]["expected"], "modes": {}}
        for seconds in (6, 30):
            started = time.perf_counter()
            _, duration, scores, _, clip_scores = analyze(predictor, path, 0, seconds)
            elapsed = time.perf_counter() - started
            top = np.argsort(-scores)[:3]
            case["modes"][str(seconds)] = {
                "predicted": GENRES[top[0]], "windows": len(clip_scores),
                "top3": {GENRES[index]: float(scores[index]) for index in top},
                "window_predictions": [GENRES[index] for index in clip_scores.argmax(axis=1)],
                "elapsed_seconds": elapsed,
            }
        assert case["modes"]["30"]["predicted"] == selected[key]["predicted"]
        case["duration"] = duration
        verified[key] = case
        if len(verified) == len(selected):
            break
    if verified.keys() != selected.keys():
        raise ValueError("В источнике отсутствуют выбранные тестовые записи.")
    examples = [verified[key] for key in selected]
    (artifacts / "demo.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(json.dumps(examples, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "genres.tar.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "demo")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    make_demo(args.dataset, args.output, args.artifacts)
