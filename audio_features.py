"""Одинаковая предобработка для обучения и приложения."""

from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import fft, signal

GENRES = (
    "blues", "classical", "country", "disco", "hiphop", "jazz",
    "metal", "pop", "reggae", "rock",
)
GENRES_RU = (
    "Блюз", "Классика", "Кантри", "Диско", "Хип-хоп", "Джаз",
    "Метал", "Поп", "Регги", "Рок",
)
ROOT = Path(__file__).resolve().parent
RATE = 16_000
SECONDS = 6
CLIP = RATE * SECONDS
MAX_CLIPS = 5
N_FFT = 1024
HOP = 320
MELS = 64
FRAMES = 1 + (CLIP - N_FFT) // HOP


def read_audio(source, start=0, seconds=SECONDS * MAX_CLIPS):
    """Читает только нужный интервал, сводит каналы и пересчитывает частоту."""
    with sf.SoundFile(source) as file:
        rate = file.samplerate
        duration = file.frames / rate
        file.seek(min(round(start * rate), file.frames))
        wave = file.read(round(seconds * rate), dtype="float32", always_2d=True)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError("В выбранном интервале нет корректного звукового сигнала.")
    wave = wave.mean(axis=1)
    if rate != RATE:
        divisor = gcd(rate, RATE)
        wave = signal.resample_poly(wave, RATE // divisor, rate // divisor)
    return wave, duration


def split_clips(wave):
    """Берёт до пяти полных окон. Единственное короткое окно дополняет нулями."""
    if len(wave) < CLIP:
        return np.pad(wave, (0, CLIP - len(wave)))[None, :]
    count = min(len(wave) // CLIP, MAX_CLIPS)
    return wave[: count * CLIP].reshape(count, CLIP)


@lru_cache(maxsize=1)
def mel_filters():
    mel_top = 2595 * np.log10(1 + RATE / 2 / 700)
    edges = 700 * (10 ** (np.linspace(0, mel_top, MELS + 2) / 2595) - 1)
    bins = fft.rfftfreq(N_FFT, 1 / RATE)
    rising = (bins - edges[:-2, None]) / np.diff(edges)[:-1, None]
    falling = (edges[2:, None] - bins) / np.diff(edges)[1:, None]
    return np.maximum(0, np.minimum(rising, falling)).astype("float32")


def log_mel(clip):
    frames = np.lib.stride_tricks.sliding_window_view(clip, N_FFT)[::HOP]
    spectrum = fft.rfft(frames * signal.windows.hann(N_FFT, sym=False).astype("float32"))
    power = np.abs(spectrum) ** 2
    mel = mel_filters() @ power.T
    decibels = 10 * np.log10(np.maximum(mel, 1e-10))
    return np.maximum(decibels - decibels.max(), -75)


def augment(clip, rng):
    """Умеренное изменение скорости, сдвиг с нулевым заполнением и шум."""
    speed = rng.uniform(0.92, 1.08)
    positions = np.arange(CLIP)
    sample_at = (positions - CLIP / 2) * speed + CLIP / 2
    wave = np.interp(sample_at, positions, clip, left=0, right=0).astype("float32")
    shift = int(rng.integers(-RATE // 4, RATE // 4 + 1))
    if shift > 0:
        wave = np.pad(wave[:-shift], (shift, 0))
    elif shift < 0:
        wave = np.pad(wave[-shift:], (0, -shift))
    rms = np.sqrt(np.mean(wave ** 2))
    snr_db = rng.uniform(18, 32)
    return wave + rng.standard_normal(CLIP, dtype=np.float32) * (rms * 10 ** (-snr_db / 20))
