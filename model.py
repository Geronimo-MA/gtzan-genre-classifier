"""Свёрточная сеть и единый интерфейс инференса."""

import numpy as np
import torch
from torch import nn

from audio_features import GENRES, ROOT, log_mel, split_clips

MODEL_PATH = ROOT / "artifacts" / "genre_cnn.pt"


class GenreCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=(2, 3), padding=2, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(96 * 4, len(GENRES)),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def normalized_tensor(features, mean, scale):
    values = (features.astype("float32") - mean) / scale
    return torch.from_numpy(values[:, None])


class Predictor:
    def __init__(self, path=MODEL_PATH):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(checkpoint["genres"]) != GENRES or checkpoint["version"] != 1:
            raise ValueError("Файл модели несовместим с этой версией приложения.")
        self.model = GenreCNN().eval()
        self.model.load_state_dict(checkpoint["state"])
        self.mean = checkpoint["mean"].numpy()
        self.scale = checkpoint["scale"].numpy()

    @torch.inference_mode()
    def predict(self, wave):
        clips = split_clips(wave)
        features = np.stack([log_mel(clip) for clip in clips])
        values = normalized_tensor(features, self.mean, self.scale)
        scores = self.model(values).softmax(dim=1).numpy()
        return scores.mean(axis=0), features, scores
