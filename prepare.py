"""Готовит признаки из оригинального архива GTZAN или каталога с записями."""

import argparse
import io
import json
import re
import tarfile
from zlib import crc32
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audio_features import GENRES, ROOT, augment, log_mel, read_audio, split_clips

SPLITS = ("train", "valid", "test")
TRACK_PATTERN = re.compile(r"([a-z]+)\.\d{5}\.(au|wav)", re.IGNORECASE)
DATA = ROOT / "data"
OUTPUT = ROOT / "artifacts"


def track_id(filename):
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if TRACK_PATTERN.fullmatch(name):
        return name.rsplit(".", 1)[0]
    return None


def split_membership(folder):
    membership = {}
    for split in SPLITS:
        names = (folder / f"{split}_filtered.txt").read_text(encoding="utf-8-sig").splitlines()
        for name in names:
            if not name.strip():
                continue
            key = track_id(name.strip())
            if key is None or key.split(".")[0] not in GENRES:
                raise ValueError(f"Некорректная запись в списке {split}: {name}")
            if key in membership:
                raise ValueError(f"Повтор записи в списках: {key}")
            membership[key] = split
    return membership


def sources(source, wanted):
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            key = track_id(path.name)
            if key in wanted and path.is_file():
                yield key, path
    else:
        with tarfile.open(source, "r|gz") as archive:
            for entry in archive:
                key = track_id(entry.name)
                if key in wanted and entry.isfile():
                    with archive.extractfile(entry) as stream:
                        yield key, io.BytesIO(stream.read())


def prepare(dataset, splits, output):
    membership = split_membership(splits)
    rows = {split: {} for split in SPLITS}
    failures = {}
    seen = set()
    with threadpool_limits(limits=1, user_api="blas"):
        for key, source in sources(dataset, membership):
            if key in seen:
                raise ValueError(f"Повтор аудиофайла в данных: {key}")
            seen.add(key)
            split = membership[key]
            try:
                wave, duration = read_audio(source)
            except (RuntimeError, ValueError, OSError) as error:
                failures[key] = str(error)
                continue
            clips = split_clips(wave)
            clean = np.stack([log_mel(clip) for clip in clips]).astype("float16")
            augmented = None
            if split == "train":
                rng = np.random.default_rng([42, crc32(key.encode("ascii"))])
                augmented = np.stack([log_mel(augment(clip, rng)) for clip in clips]).astype("float16")
            rows[split][key] = (duration, clean, augmented)
            if len(seen) % 100 == 0:
                print(f"Processed {len(seen)}/{len(membership)} tracks", flush=True)
    missing = sorted(membership.keys() - seen)
    if missing:
        raise ValueError(f"В источнике нет {len(missing)} ожидаемых записей: {missing[:10]}")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"source": dataset.name, "failures": failures, "partitions": {}}
    for split in SPLITS:
        keys = sorted(rows[split])
        track_labels = [GENRES.index(key.split(".")[0]) for key in keys]
        counts = np.bincount(track_labels, minlength=len(GENRES))
        if not counts.all():
            raise ValueError(f"В выборке {split} отсутствует один или несколько жанров.")
        durations, clean, augmented = zip(*(rows[split][key] for key in keys))
        data = {
            "x": np.concatenate(clean),
            "y": np.repeat(track_labels, [len(item) for item in clean]),
            "track": np.repeat(keys, [len(item) for item in clean]),
        }
        if split == "train":
            data["augmented"] = np.concatenate(augmented)
        np.savez(output / f"{split}.npz", **data)
        summary["partitions"][split] = {
            "tracks": len(keys), "clips": len(data["y"]),
            "class_counts": dict(zip(GENRES, counts.tolist())),
            "duration_min": float(min(durations)),
            "duration_mean": float(np.mean(durations)),
            "duration_max": float(max(durations)),
        }
        print(f"{split}: {len(keys)} tracks, {len(data['y'])} clips", flush=True)
    (output / "dataset.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATA / "genres.tar.gz")
    parser.add_argument("--splits", type=Path, default=DATA / "splits")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.dataset, args.splits, args.output)
