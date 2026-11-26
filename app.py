"""Локальное приложение для классификации музыки."""

import argparse
import json
import queue
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import soundfile as sf
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from threadpoolctl import threadpool_limits

from audio_features import GENRES, GENRES_RU, HOP, N_FFT, RATE, ROOT, SECONDS, read_audio
from model import Predictor

BG = "#101722"
PANEL = "#182231"
LINE = "#2a3a4e"
TEXT = "#f2f6fb"
MUTED = "#a3b3c6"
ACCENT = "#64e0c3"
DEMOS = ROOT / "demo"


def analyze(predictor, path, start, seconds):
    wave, duration = read_audio(path, start=start, seconds=seconds)
    if len(wave) < RATE * 2:
        raise ValueError("Выбери фрагмент длительностью не меньше двух секунд.")
    if np.sqrt(np.mean(wave ** 2)) < 1e-5:
        raise ValueError("В выбранном фрагменте почти нет звука.")
    with threadpool_limits(limits=1, user_api="blas"):
        scores, features, clip_scores = predictor.predict(wave)
    return wave, duration, scores, features, clip_scores


class GenreApp:
    def __init__(self, root, predictor):
        self.root = root
        self.predictor = predictor
        self.path = None
        self.wave = None
        self.results = queue.Queue()
        self.demos = sorted(DEMOS.glob("*.wav"))
        self.audio_temp = tempfile.TemporaryDirectory(prefix="genre_audio_")
        self.audio_path = Path(self.audio_temp.name) / "selection.wav"
        self.start = tk.StringVar(value="0")
        self.mode = tk.StringVar(value="6 секунд")
        self.status = tk.StringVar(value="Открой аудиофайл или выбери тестовую запись.")
        root.title("Классификатор музыкальных жанров")
        root.configure(bg=BG)
        root.geometry("1100x720")
        root.minsize(850, 645)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.build()

    def label(self, parent, text="", size=10, color=TEXT, bold=False, **kwargs):
        font = ("Segoe UI", size, "bold" if bold else "normal")
        return tk.Label(parent, text=text, font=font, fg=color, bg=parent["bg"], **kwargs)

    def button(self, parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, relief="flat",
                         font=("Segoe UI", 10, "bold"), padx=15, pady=7,
                         bg=ACCENT if primary else LINE, fg=BG if primary else TEXT,
                         activebackground="#b7f4e5" if primary else "#40536a",
                         activeforeground=BG if primary else TEXT, borderwidth=0,
                         disabledforeground="#7f8da0", cursor="hand2")

    def build(self):
        header = tk.Frame(self.root, bg=BG, padx=24, pady=15)
        header.pack(fill="x")
        self.label(header, "Классификация музыкальных жанров", 23, bold=True).pack(anchor="w")
        self.label(header, "GTZAN  ·  10 жанров  ·  CNN на log-mel признаках", 10, MUTED).pack(anchor="w")

        controls = tk.Frame(self.root, bg=PANEL, padx=14, pady=12)
        controls.pack(fill="x", padx=24)
        self.open_button = self.button(controls, "Открыть аудио", self.pick, primary=True)
        self.open_button.pack(side="left", padx=(0, 12))
        self.demo_menu = ttk.Combobox(controls, values=[p.stem for p in self.demos],
                                      width=20, state="readonly" if self.demos else "disabled")
        self.demo_menu.set("Тестовая запись")
        self.demo_menu.pack(side="left", padx=(0, 14))
        self.demo_menu.bind("<<ComboboxSelected>>", self.pick_demo)
        self.label(controls, "Начало, с", 10, MUTED).pack(side="left", padx=(0, 6))
        self.start_entry = ttk.Entry(controls, width=6, textvariable=self.start)
        self.start_entry.pack(side="left", padx=(0, 12))
        self.mode_menu = ttk.Combobox(controls, textvariable=self.mode, width=16,
                                      state="readonly", values=("6 секунд", "До 30 секунд"))
        self.mode_menu.pack(side="left", padx=(0, 10))
        self.go_button = self.button(controls, "Анализировать", self.run)
        self.go_button.pack(side="left")

        self.file_text = self.label(self.root, "Файл не выбран", 10, MUTED, anchor="w")
        self.file_text.pack(fill="x", padx=24, pady=(7, 7))
        body = tk.Frame(self.root, bg=BG)
        body.pack(expand=True, fill="both", padx=24)
        graph_panel = tk.Frame(body, bg=PANEL)
        graph_panel.pack(side="left", expand=True, fill="both", padx=(0, 14))
        self.figure = Figure(figsize=(6, 4.5), facecolor=PANEL, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_panel)
        self.canvas.get_tk_widget().pack(expand=True, fill="both", padx=8, pady=8)
        self.draw_plots()

        result_panel = tk.Frame(body, bg=PANEL, padx=16, pady=12, width=240)
        result_panel.pack(side="right", fill="y")
        self.label(result_panel, "Предсказанный жанр", 11, MUTED).pack(anchor="w")
        self.winner = self.label(result_panel, "—", 24, ACCENT, bold=True)
        self.winner.pack(anchor="w", pady=(0, 5))
        self.score_rows = []
        for _ in GENRES:
            row = tk.Frame(result_panel, bg=PANEL)
            row.pack(fill="x", pady=2)
            name = self.label(row, "—", 9, MUTED, anchor="w", width=10)
            name.pack(side="left")
            bar = tk.Canvas(row, width=88, height=6, bg=LINE, highlightthickness=0)
            bar.pack(side="left", padx=(0, 6))
            score = self.label(row, "—", 9, MUTED, anchor="e", width=6)
            score.pack(side="left")
            self.score_rows.append((name, bar, score))
        self.label(result_panel, "Оценки softmax", 9, MUTED).pack(anchor="w", pady=(5, 0))
        self.demo_truth = self.label(result_panel, "", 9, MUTED, anchor="w")
        self.demo_truth.pack(fill="x", pady=(7, 0))

        footer = tk.Frame(self.root, bg=BG, padx=24, pady=10)
        footer.pack(fill="x")
        self.play_button = self.button(footer, "Прослушать", self.play)
        self.play_button.pack(side="left", padx=(0, 6))
        self.play_button.configure(state="disabled")
        self.stop_button = self.button(footer, "Стоп", self.stop_audio)
        self.stop_button.pack(side="left", padx=(0, 12))
        self.label(footer, textvariable=self.status, color=MUTED).pack(side="left")

    def draw_plots(self, features=None, start=0):
        self.figure.clear()
        wave_ax, mel_ax = self.figure.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]})
        for ax, title, ylabel in (
            (wave_ax, "Звуковая волна", "Амплитуда"),
            (mel_ax, "Log-mel спектрограмма первого окна", "Mel-полоса"),
        ):
            ax.set(facecolor=BG, xlabel="Время, с", ylabel=ylabel)
            ax.set_title(title, loc="left", color=TEXT, fontsize=10)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.xaxis.label.set_color(MUTED)
            ax.yaxis.label.set_color(MUTED)
            for spine in ax.spines.values():
                spine.set_color(LINE)
        if features is not None:
            step = max(len(self.wave) // 6000, 1)
            time_axis = start + np.arange(0, len(self.wave), step) / RATE
            wave_ax.plot(time_axis, self.wave[::step], color=ACCENT, linewidth=0.7)
            extent = start + (features.shape[-1] * HOP + N_FFT - HOP) / RATE
            mel_ax.imshow(features[0], origin="lower", aspect="auto", cmap="magma",
                          extent=(start, extent, 0, features.shape[1]), vmin=-75, vmax=0)
        else:
            for ax in (wave_ax, mel_ax):
                ax.set_xticks([])
                ax.set_yticks([])
        self.canvas.draw_idle()

    def pick(self):
        file = filedialog.askopenfilename(filetypes=[
            ("Аудио", "*.wav *.flac *.ogg *.mp3 *.au *.aiff"), ("Все файлы", "*.*"),
        ])
        if file:
            self.demo_menu.set("Тестовая запись")
            self.open_file(Path(file))

    def pick_demo(self, _event):
        self.open_file(self.demos[self.demo_menu.current()])

    def open_file(self, path):
        self.path = path
        self.start.set("0")
        self.run()

    def run(self):
        if self.path is None:
            self.status.set("Сначала открой аудиофайл.")
            return
        try:
            start = float(self.start.get().replace(",", "."))
            if not np.isfinite(start) or start < 0:
                raise ValueError
        except ValueError:
            self.status.set("Начало должно быть неотрицательным числом.")
            return
        self.stop_audio()
        self.clear_result()
        self.file_text.configure(text=self.path.name)
        self.status.set("Анализ аудио…")
        self.set_busy(True)
        seconds = SECONDS if self.mode.get() == "6 секунд" else 30
        threading.Thread(target=self.worker, args=(self.path, start, seconds), daemon=True).start()
        self.root.after(40, self.poll)

    def worker(self, path, start, seconds):
        try:
            started = time.perf_counter()
            result = analyze(self.predictor, path, start, seconds)
            self.results.put((start, time.perf_counter() - started, result, None))
        except Exception as error:
            self.results.put((start, 0, None, error))

    def set_busy(self, busy):
        self.open_button.configure(state="disabled" if busy else "normal")
        self.go_button.configure(state="disabled" if busy else "normal")
        self.demo_menu.configure(state="disabled" if busy or not self.demos else "readonly")

    def clear_result(self):
        self.wave = None
        self.winner.configure(text="—")
        self.demo_truth.configure(text="")
        self.play_button.configure(state="disabled")
        for name, bar, label in self.score_rows:
            name.configure(text="—", fg=MUTED)
            label.configure(text="—")
            bar.delete("all")
        self.draw_plots()

    def poll(self):
        try:
            start, elapsed, result, error = self.results.get_nowait()
        except queue.Empty:
            self.root.after(40, self.poll)
            return
        self.set_busy(False)
        if error is not None:
            self.status.set("Не удалось проанализировать аудио.")
            messagebox.showerror("Ошибка чтения или анализа", str(error))
            return
        self.wave, duration, scores, features, clip_scores = result
        indices = np.argsort(-scores)
        self.winner.configure(text=GENRES_RU[indices[0]])
        if self.path in self.demos:
            genre = self.path.stem.split(".")[0]
            self.demo_truth.configure(text=f"GTZAN · {GENRES_RU[GENRES.index(genre)]}")
        for position, (index, (name, bar, label)) in enumerate(zip(indices, self.score_rows)):
            color = ACCENT if position == 0 else "#6f9ec5"
            name.configure(text=GENRES_RU[index], fg=TEXT)
            bar.create_rectangle(0, 0, round(scores[index] * 88), 6, fill=color, width=0)
            label.configure(text=f"{scores[index]:.1%}")
        self.draw_plots(features, start)
        self.file_text.configure(text=f"{self.path.name}   ·   {duration:.1f} с")
        self.status.set(f"От {start:.1f} с   ·   Окон: {len(clip_scores)}   ·   Анализ: {elapsed:.2f} с")
        self.play_button.configure(state="normal")

    def play(self):
        import winsound

        if self.wave is not None:
            self.stop_audio()
            sf.write(self.audio_path, self.wave, RATE, subtype="PCM_16")
            winsound.PlaySound(str(self.audio_path), winsound.SND_FILENAME | winsound.SND_ASYNC)

    def stop_audio(self):
        import winsound

        winsound.PlaySound(None, 0)

    def close(self):
        self.stop_audio()
        self.audio_temp.cleanup()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="Путь к аудиофайлу")
    parser.add_argument("--cli", action="store_true", help="Вывести результат JSON")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=int, choices=[6, 30], default=6)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.cli:
        if args.file is None or not np.isfinite(args.start) or args.start < 0:
            parser.error("Режим --cli требует аудиофайл и неотрицательное начало.")
        try:
            _, duration, scores, _, clips = analyze(Predictor(), args.file, args.start, args.seconds)
        except (OSError, RuntimeError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        print(json.dumps({
            "file": args.file.name, "genre": GENRES[int(scores.argmax())],
            "duration": duration, "windows": len(clips),
            "scores": {key: float(score) for key, score in zip(GENRES, scores)},
        }, indent=2))
        return
    root = tk.Tk()
    try:
        app = GenreApp(root, Predictor())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        root.withdraw()
        messagebox.showerror("Не удалось загрузить модель", str(error))
        root.destroy()
        return
    if args.file:
        root.after(50, app.open_file, args.file)
    root.mainloop()


if __name__ == "__main__":
    main()
