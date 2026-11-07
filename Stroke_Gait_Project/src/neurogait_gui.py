
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

SRC = Path(__file__).resolve().parent
PD_SRC = SRC / "parkinson"
if str(PD_SRC) not in sys.path:
    sys.path.insert(0, str(PD_SRC))

from predict import predict_gait
from pd_final_predict import predict_signal

BG = "#EDF2F8"
NAVY = "#16324F"
BLUE = "#2668C7"
GREEN = "#14845A"
RED = "#C44953"
TEXT = "#26394A"
MUTED = "#62758A"
WHITE = "#FFFFFF"
BORDER = "#D5E1ED"

root = tk.Tk()
root.title("NeuroGait | Bilateral IMU Gait Screening")
root.geometry("1090x780")
root.minsize(870, 650)
root.configure(bg=BG)

style = ttk.Style(root)
style.theme_use("clam")
style.configure("NG.TNotebook", background=BG, borderwidth=0)
style.configure("NG.TNotebook.Tab", padding=(22, 12), font=("Segoe UI", 11, "bold"))
style.map("NG.TNotebook.Tab", background=[("selected", WHITE)], foreground=[("selected", BLUE)])
style.configure("Healthy.Horizontal.TProgressbar", background=GREEN, troughcolor="#E4EBF1", thickness=15)
style.configure("Risk.Horizontal.TProgressbar", background=RED, troughcolor="#E4EBF1", thickness=15)

status = tk.StringVar(value="Ready. Select an input matching one of the two trained datasets.")


def text(parent, value, size=10, bold=False, colour=TEXT, bg=WHITE, **kwargs):
    return tk.Label(parent, text=value, bg=bg, fg=colour,
                    font=("Segoe UI", size, "bold" if bold else "normal"), **kwargs)


def card(parent, **kwargs):
    return tk.Frame(parent, bg=WHITE, highlightbackground=BORDER,
                    highlightthickness=1, **kwargs)


def choose_file(var):
    selected = filedialog.askopenfilename(
        title="Select IMU recording", filetypes=[("CSV files", "*.csv")]
    )
    if selected:
        var.set(selected)


def file_row(parent, label, var):
    row = tk.Frame(parent, bg=WHITE)
    row.pack(fill="x", pady=7)
    text(row, label, size=10, bold=True, width=19, anchor="w").pack(side="left")
    tk.Entry(row, textvariable=var, font=("Segoe UI", 9),
             relief="solid", bd=1).pack(side="left", fill="x", expand=True, ipady=7)
    tk.Button(row, text="Browse", command=lambda: choose_file(var),
              bg=BLUE, fg=WHITE, relief="flat", cursor="hand2",
              font=("Segoe UI", 10, "bold"), padx=14, pady=6).pack(side="left", padx=(10, 0))


header = tk.Frame(root, bg=NAVY, height=105)
header.pack(fill="x")
header.pack_propagate(False)
text(header, "NeuroGait", 27, True, WHITE, NAVY).pack(anchor="w", padx=30, pady=(15, 0))
text(header, "Bilateral foot IMU analysis | Separate Stroke and Parkinson's models",
     11, False, "#C6D9E9", NAVY).pack(anchor="w", padx=32, pady=(0, 12))

body = tk.Frame(root, bg=BG)
body.pack(fill="both", expand=True, padx=25, pady=17)
notebook = ttk.Notebook(body, style="NG.TNotebook")
notebook.pack(fill="both", expand=True)

stroke_tab = tk.Frame(notebook, bg=BG)
pd_tab = tk.Frame(notebook, bg=BG)
notebook.add(stroke_tab, text="Stroke screening")
notebook.add(pd_tab, text="Parkinson's screening")


def make_tab(tab, description, file_specs, on_analyse):
    info = card(tab)
    info.pack(fill="x", pady=(13, 10))
    text(info, description, size=10, colour=MUTED, wraplength=970,
         justify="left").pack(anchor="w", padx=20, pady=13)

    upload = card(tab)
    upload.pack(fill="x", pady=(0, 10))
    text(upload, "Input gait recording", 14, True).pack(anchor="w", padx=20, pady=(15, 4))
    fields = tk.Frame(upload, bg=WHITE)
    fields.pack(fill="x", padx=20, pady=(0, 12))
    variables = []
    for label in file_specs:
        var = tk.StringVar()
        file_row(fields, label, var)
        variables.append(var)

    actions = tk.Frame(tab, bg=BG)
    actions.pack(fill="x", pady=(2, 10))
    button = tk.Button(actions, text="Analyse gait recording", bg=BLUE, fg=WHITE,
                       relief="flat", font=("Segoe UI", 11, "bold"),
                       cursor="hand2", padx=21, pady=10)
    button.pack(side="left")

    results = card(tab)
    results.pack(fill="both", expand=True)
    text(results, "SCREENING RESULT", 10, True, MUTED).pack(anchor="w", padx=22, pady=(20, 2))
    prediction = tk.StringVar(value="Not analysed yet")
    prediction_label = text(results, "", 21, True, MUTED, wraplength=900, justify="left")
    prediction_label.configure(textvariable=prediction)
    prediction_label.pack(anchor="w", padx=22, pady=(4, 12))

    score_a = tk.StringVar(value="--")
    score_b = tk.StringVar(value="--")
    score_rows = tk.Frame(results, bg=WHITE)
    score_rows.pack(fill="x", padx=22, pady=(3, 10))
    bar_a = ttk.Progressbar(score_rows, maximum=100, style="Healthy.Horizontal.TProgressbar")
    bar_b = ttk.Progressbar(score_rows, maximum=100, style="Risk.Horizontal.TProgressbar")

    for label, variable, bar in (("Healthy-associated score", score_a, bar_a),
                                 ("Condition-associated score", score_b, bar_b)):
        line = tk.Frame(score_rows, bg=WHITE)
        line.pack(fill="x", pady=(4, 2))
        text(line, label, 10, True).pack(side="left")
        text(line, "", 11, True, BLUE, textvariable=variable).pack(side="right")
        bar.pack(fill="x", pady=(1, 8))

    windows_var = tk.StringVar(value="Analysed windows: --")
    text(results, "", 10, False, MUTED, textvariable=windows_var).pack(anchor="w", padx=22, pady=6)
    text(results, "Model scores are not calibrated disease probabilities. Research screening only; not a medical diagnosis.",
         9, False, MUTED, wraplength=900, justify="left").pack(anchor="w", padx=22, pady=(8, 18))

    def reset():
        for var in variables:
            var.set("")
        prediction.set("Not analysed yet")
        prediction_label.configure(fg=MUTED)
        score_a.set("--")
        score_b.set("--")
        bar_a["value"] = 0
        bar_b["value"] = 0
        windows_var.set("Analysed windows: --")
        status.set("Ready.")

    tk.Button(actions, text="Clear", command=reset, bg="#DCE6F0", fg=TEXT,
              relief="flat", font=("Segoe UI", 10, "bold"),
              padx=18, pady=10).pack(side="left", padx=12)

    def finish(result=None, error=None):
        button.configure(state="normal", text="Analyse gait recording", bg=BLUE)
        if error is not None:
            status.set("Analysis failed. Check the error and CSV format.")
            messagebox.showerror("NeuroGait analysis error", str(error))
            return
        condition = float(result["condition_score"])
        healthy = float(result["healthy_score"])
        prediction.set(result["prediction"])
        prediction_label.configure(fg=RED if condition >= 50 else GREEN)
        score_a.set(f"{healthy:.2f}%")
        score_b.set(f"{condition:.2f}%")
        bar_a["value"] = healthy
        bar_b["value"] = condition
        windows_var.set(f"Analysed windows: {result['windows']}")
        status.set("Analysis completed. This is a research screening result.")

    def start():
        paths = [var.get().strip() for var in variables]
        if any(not p for p in paths):
            messagebox.showwarning("Missing recording", "Select every required CSV file first.")
            return
        if any(not Path(p).is_file() for p in paths):
            messagebox.showerror("File not found", "One or more selected files do not exist.")
            return
        button.configure(state="disabled", text="Analysing...", bg="#84A9DC")
        status.set("Analysing input using the selected model...")
        # Heavy inference is performed off the UI thread; Tk updates stay on the UI thread.
        def worker():
            try:
                output = on_analyse(*paths)
            except Exception as exc:
                root.after(0, lambda error=exc: finish(error=error))
            else:
                root.after(0, lambda output=output: finish(result=output))
        threading.Thread(target=worker, daemon=True).start()

    button.configure(command=start)


def analyse_stroke(left, right):
    result = predict_gait(Path(left), Path(right))
    return {
        "prediction": result["prediction"],
        "condition_score": result["stroke_probability"],
        "healthy_score": result["healthy_probability"],
        "windows": result["windows"],
    }


def analyse_pd(csv_path):
    df = pd.read_csv(csv_path)
    required = {"foot", "timestamp", "acx", "acy", "acz", "gyrx", "gyry", "gyrz"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Wrong PD CSV format. Missing columns: " + ", ".join(missing))
    feet = set(df["foot"].dropna().astype(str).str.strip().str.lower().unique())
    if not {"left", "right"}.issubset(feet):
        raise ValueError("PD CSV must contain both 'left' and 'right' rows in the foot column.")
    output = predict_signal(df)
    pd_score = output["pd_probability"] * 100
    return {
        "prediction": output["prediction"],
        "condition_score": pd_score,
        "healthy_score": 100 - pd_score,
        "windows": output["windows"],
    }


make_tab(stroke_tab,
         "Stroke model: two separate left/right foot CSV files, matching the original 104 Hz Stroke dataset. Uses the existing Stroke CNN-LSTM unchanged.",
         ["Left foot CSV", "Right foot CSV"], analyse_stroke)
make_tab(pd_tab,
         "Parkinson's model: one combined CSV containing left and right foot measurements (36 Hz resampling). Uses the final multi-task CNN-LSTM checkpoint.",
         ["Combined PD CSV"], analyse_pd)

footer = tk.Frame(root, bg=NAVY, height=36)
footer.pack(side="bottom", fill="x")
footer.pack_propagate(False)
text(footer, "", 9, False, WHITE, NAVY, textvariable=status).pack(side="left", padx=18, pady=8)

root.mainloop()
