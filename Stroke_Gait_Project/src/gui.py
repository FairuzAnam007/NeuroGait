from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from predict import predict_gait


BG = "#EEF3F8"
WHITE = "#FFFFFF"

NAVY = "#173653"
NAVY_2 = "#234B6D"

BLUE = "#2F80ED"
BLUE_HOVER = "#246BC7"
BLUE_LIGHT = "#EAF3FF"

GREEN = "#1FA66A"
GREEN_LIGHT = "#EAF8F1"

RED = "#E45161"
RED_LIGHT = "#FDECEF"

TEXT = "#213547"
MUTED = "#738395"
BORDER = "#D7E2EC"


root = tk.Tk()

root.title("Stroke Gait Classification System")

# Enough height for all result components
root.geometry("1200x850")

root.minsize(1100, 800)

root.configure(bg=BG)


left_file_path = tk.StringVar()
right_file_path = tk.StringVar()

prediction_var = tk.StringVar(
    value="Not analysed yet"
)

healthy_prob_var = tk.StringVar(
    value="--"
)

stroke_prob_var = tk.StringVar(
    value="--"
)

confidence_var = tk.StringVar(
    value="--"
)

windows_var = tk.StringVar(
    value="--"
)

status_var = tk.StringVar(
    value="Ready for analysis"
)


style = ttk.Style()
style.theme_use("clam")

style.configure(
    "Healthy.Horizontal.TProgressbar",
    troughcolor="#DDECE5",
    background=GREEN,
    bordercolor="#DDECE5",
    lightcolor=GREEN,
    darkcolor=GREEN,
    thickness=16
)

style.configure(
    "Stroke.Horizontal.TProgressbar",
    troughcolor="#F2DDE1",
    background=RED,
    bordercolor="#F2DDE1",
    lightcolor=RED,
    darkcolor=RED,
    thickness=16
)

def browse_left():

    file_path = filedialog.askopenfilename(
        title="Select Left Foot CSV",
        filetypes=[
            ("CSV files", "*.csv")
        ]
    )

    if file_path:

        left_file_path.set(
            str(Path(file_path).resolve())
        )

        left_status.config(
            text="✓ Left recording selected",
            fg=GREEN
        )

        status_var.set(
            "Left foot recording selected"
        )


def browse_right():

    file_path = filedialog.askopenfilename(
        title="Select Right Foot CSV",
        filetypes=[
            ("CSV files", "*.csv")
        ]
    )

    if file_path:

        right_file_path.set(
            str(Path(file_path).resolve())
        )

        right_status.config(
            text="✓ Right recording selected",
            fg=GREEN
        )

        status_var.set(
            "Right foot recording selected"
        )


def clear_all():

    left_file_path.set("")
    right_file_path.set("")

    prediction_var.set(
        "Not analysed yet"
    )

    healthy_prob_var.set("--")
    stroke_prob_var.set("--")
    confidence_var.set("--")
    windows_var.set("--")

    healthy_bar["value"] = 0
    stroke_bar["value"] = 0

    left_status.config(
        text="No file selected",
        fg=MUTED
    )

    right_status.config(
        text="No file selected",
        fg=MUTED
    )

    prediction_label.config(
        fg=MUTED
    )

    result_badge.config(
        text="● WAITING FOR ANALYSIS",
        fg=MUTED
    )

    result_card.config(
        highlightbackground=BORDER
    )

    status_var.set(
        "Ready for analysis"
    )


def analyse_gait():

    left_text = left_file_path.get().strip()
    right_text = right_file_path.get().strip()


    if not left_text:

        messagebox.showerror(
            "Missing File",
            "Please select the Left Foot CSV file."
        )

        return


    if not right_text:

        messagebox.showerror(
            "Missing File",
            "Please select the Right Foot CSV file."
        )

        return


    left_file = Path(left_text)
    right_file = Path(right_text)


    if not left_file.exists():

        messagebox.showerror(
            "File Not Found",
            f"Left Foot file was not found:\n\n{left_file}"
        )

        return


    if not right_file.exists():

        messagebox.showerror(
            "File Not Found",
            f"Right Foot file was not found:\n\n{right_file}"
        )

        return


    try:

        analyse_button.config(
            text="Analysing...",
            state="disabled",
            bg="#8EB9EA"
        )

        clear_button.config(
            state="disabled"
        )

        status_var.set(
            "Analysing bilateral IMU gait data..."
        )

        root.update()


        # -------------------------------
        # MODEL PREDICTION
        # -------------------------------

        result = predict_gait(
            left_file,
            right_file
        )


        prediction = result[
            "prediction"
        ]

        healthy = result[
            "healthy_probability"
        ]

        stroke = result[
            "stroke_probability"
        ]

        confidence = result[
            "confidence"
        ]

        windows = result[
            "windows"
        ]


        prediction_var.set(
            prediction
        )

        healthy_prob_var.set(
            f"{healthy:.2f}%"
        )

        stroke_prob_var.set(
            f"{stroke:.2f}%"
        )

        confidence_var.set(
            f"{confidence:.2f}%"
        )

        windows_var.set(
            str(windows)
        )


        healthy_bar["value"] = healthy
        stroke_bar["value"] = stroke



        if prediction == "Healthy gait":

            prediction_label.config(
                fg=GREEN
            )

            result_badge.config(
                text="● HEALTHY GAIT PATTERN",
                fg=GREEN
            )

            result_card.config(
                highlightbackground=GREEN
            )

        else:

            prediction_label.config(
                fg=RED
            )

            result_badge.config(
                text="● STROKE-ASSOCIATED GAIT PATTERN",
                fg=RED
            )

            result_card.config(
                highlightbackground=RED
            )


        status_var.set(
            "Analysis completed successfully"
        )


    except Exception as error:

        messagebox.showerror(
            "Analysis Error",
            str(error)
        )

        status_var.set(
            "Analysis failed"
        )


    finally:

        analyse_button.config(
            text="Analyse Gait",
            state="normal",
            bg=BLUE
        )

        clear_button.config(
            state="normal"
        )


header = tk.Frame(
    root,
    bg=NAVY,
    height=100
)

header.pack(
    fill="x"
)

header.pack_propagate(False)


tk.Label(
    header,
    text="STROKE GAIT CLASSIFICATION SYSTEM",
    font=(
        "Segoe UI",
        26,
        "bold"
    ),
    fg="white",
    bg=NAVY
).pack(
    pady=(17, 2)
)


tk.Label(
    header,
    text="Bilateral Foot IMU-Based Deep Learning Analysis",
    font=(
        "Segoe UI",
        11
    ),
    fg="#C8DBED",
    bg=NAVY
).pack()


main = tk.Frame(
    root,
    bg=BG
)

main.pack(
    fill="both",
    expand=True,
    padx=24,
    pady=12
)


config_card = tk.Frame(
    main,
    bg=WHITE,
    highlightbackground=BORDER,
    highlightthickness=1,
    height=115
)

config_card.pack(
    fill="x",
    pady=(0, 10)
)

config_card.pack_propagate(False)


tk.Label(
    config_card,
    text="Model Configuration",
    font=(
        "Segoe UI",
        12,
        "bold"
    ),
    fg=NAVY,
    bg=WHITE
).pack(
    anchor="w",
    padx=20,
    pady=(10, 5)
)


config_row = tk.Frame(
    config_card,
    bg=WHITE
)

config_row.pack(
    fill="x",
    padx=15
)


config_data = [

    ("MODEL", "CNN-LSTM"),

    ("SAMPLING RATE", "104 Hz"),

    ("WINDOW", "5 seconds"),

    ("OVERLAP", "50%"),

    ("INPUT CHANNELS", "12")

]


for title, value in config_data:

    box = tk.Frame(
        config_row,
        bg=BLUE_LIGHT,
        height=58
    )

    box.pack(
        side="left",
        expand=True,
        fill="both",
        padx=4
    )

    box.pack_propagate(False)


    tk.Label(
        box,
        text=title,
        font=(
            "Segoe UI",
            8
        ),
        fg=MUTED,
        bg=BLUE_LIGHT
    ).pack(
        pady=(8, 1)
    )


    tk.Label(
        box,
        text=value,
        font=(
            "Segoe UI",
            11,
            "bold"
        ),
        fg=BLUE,
        bg=BLUE_LIGHT
    ).pack()


file_card = tk.Frame(
    main,
    bg=WHITE,
    highlightbackground=BORDER,
    highlightthickness=1,
    height=175
)

file_card.pack(
    fill="x",
    pady=(0, 10)
)

file_card.pack_propagate(False)


tk.Label(
    file_card,
    text="Select IMU Recordings",
    font=(
        "Segoe UI",
        12,
        "bold"
    ),
    fg=NAVY,
    bg=WHITE
).grid(
    row=0,
    column=0,
    columnspan=3,
    sticky="w",
    padx=20,
    pady=(10, 6)
)


# LEFT FILE

tk.Label(
    file_card,
    text="Left Foot CSV",
    font=(
        "Segoe UI",
        10,
        "bold"
    ),
    fg=TEXT,
    bg=WHITE
).grid(
    row=1,
    column=0,
    sticky="w",
    padx=(20, 12),
    pady=4
)


left_entry = tk.Entry(
    file_card,
    textvariable=left_file_path,
    font=(
        "Segoe UI",
        9
    ),
    relief="solid",
    bd=1
)

left_entry.grid(
    row=1,
    column=1,
    sticky="ew",
    ipady=6
)


tk.Button(
    file_card,
    text="Browse",
    command=browse_left,
    bg=BLUE,
    fg="white",
    activebackground=BLUE_HOVER,
    activeforeground="white",
    font=(
        "Segoe UI",
        9,
        "bold"
    ),
    relief="flat",
    width=10,
    cursor="hand2"
).grid(
    row=1,
    column=2,
    padx=15,
    ipady=4
)


left_status = tk.Label(
    file_card,
    text="No file selected",
    font=(
        "Segoe UI",
        8
    ),
    fg=MUTED,
    bg=WHITE
)

left_status.grid(
    row=2,
    column=1,
    sticky="w",
    pady=(1, 5)
)


# RIGHT FILE

tk.Label(
    file_card,
    text="Right Foot CSV",
    font=(
        "Segoe UI",
        10,
        "bold"
    ),
    fg=TEXT,
    bg=WHITE
).grid(
    row=3,
    column=0,
    sticky="w",
    padx=(20, 12),
    pady=4
)


right_entry = tk.Entry(
    file_card,
    textvariable=right_file_path,
    font=(
        "Segoe UI",
        9
    ),
    relief="solid",
    bd=1
)

right_entry.grid(
    row=3,
    column=1,
    sticky="ew",
    ipady=6
)


tk.Button(
    file_card,
    text="Browse",
    command=browse_right,
    bg=BLUE,
    fg="white",
    activebackground=BLUE_HOVER,
    activeforeground="white",
    font=(
        "Segoe UI",
        9,
        "bold"
    ),
    relief="flat",
    width=10,
    cursor="hand2"
).grid(
    row=3,
    column=2,
    padx=15,
    ipady=4
)


right_status = tk.Label(
    file_card,
    text="No file selected",
    font=(
        "Segoe UI",
        8
    ),
    fg=MUTED,
    bg=WHITE
)

right_status.grid(
    row=4,
    column=1,
    sticky="w",
    pady=(1, 8)
)


file_card.columnconfigure(
    1,
    weight=1
)


button_row = tk.Frame(
    main,
    bg=BG,
    height=55
)

button_row.pack(
    fill="x",
    pady=(0, 10)
)

button_row.pack_propagate(False)


button_inner = tk.Frame(
    button_row,
    bg=BG
)

button_inner.pack()


analyse_button = tk.Button(
    button_inner,
    text="Analyse Gait",
    command=analyse_gait,
    bg=BLUE,
    fg="white",
    activebackground=BLUE_HOVER,
    activeforeground="white",
    font=(
        "Segoe UI",
        12,
        "bold"
    ),
    relief="flat",
    width=18,
    cursor="hand2"
)

analyse_button.pack(
    side="left",
    padx=7,
    ipady=8
)


clear_button = tk.Button(
    button_inner,
    text="Clear",
    command=clear_all,
    bg="#DCE5ED",
    fg=TEXT,
    activebackground="#CDD9E3",
    font=(
        "Segoe UI",
        10,
        "bold"
    ),
    relief="flat",
    width=10,
    cursor="hand2"
)

clear_button.pack(
    side="left",
    padx=7,
    ipady=8
)

# FIX:
# Result panel now has its own guaranteed height.
result_card = tk.Frame(
    main,
    bg=WHITE,
    highlightbackground=BORDER,
    highlightthickness=2,
    height=285
)

result_card.pack(
    fill="x"
)

result_card.pack_propagate(False)


classification_top = tk.Frame(
    result_card,
    bg=WHITE
)

classification_top.pack(
    fill="x",
    pady=(12, 2)
)


tk.Label(
    classification_top,
    text="FINAL CLASSIFICATION",
    font=(
        "Segoe UI",
        9,
        "bold"
    ),
    fg=MUTED,
    bg=WHITE
).pack()


prediction_label = tk.Label(
    classification_top,
    textvariable=prediction_var,
    font=(
        "Segoe UI",
        23,
        "bold"
    ),
    fg=MUTED,
    bg=WHITE
)

prediction_label.pack(
    pady=(2, 0)
)


result_badge = tk.Label(
    classification_top,
    text="● WAITING FOR ANALYSIS",
    font=(
        "Segoe UI",
        8,
        "bold"
    ),
    fg=MUTED,
    bg=WHITE
)

result_badge.pack(
    pady=(1, 8)
)


probability_row = tk.Frame(
    result_card,
    bg=WHITE
)

probability_row.pack(
    fill="x",
    padx=35,
    pady=(2, 8)
)


healthy_card = tk.Frame(
    probability_row,
    bg=GREEN_LIGHT,
    height=90,
    highlightbackground="#CDEADA",
    highlightthickness=1
)

healthy_card.pack(
    side="left",
    expand=True,
    fill="both",
    padx=(0, 8)
)

healthy_card.pack_propagate(False)


healthy_top = tk.Frame(
    healthy_card,
    bg=GREEN_LIGHT
)

healthy_top.pack(
    fill="x",
    padx=18,
    pady=(12, 5)
)


tk.Label(
    healthy_top,
    text="HEALTHY",
    font=(
        "Segoe UI",
        10,
        "bold"
    ),
    fg=GREEN,
    bg=GREEN_LIGHT
).pack(
    side="left"
)


tk.Label(
    healthy_top,
    textvariable=healthy_prob_var,
    font=(
        "Segoe UI",
        18,
        "bold"
    ),
    fg=GREEN,
    bg=GREEN_LIGHT
).pack(
    side="right"
)


healthy_bar = ttk.Progressbar(
    healthy_card,
    style="Healthy.Horizontal.TProgressbar",
    maximum=100
)

healthy_bar.pack(
    fill="x",
    padx=18,
    pady=(3, 12),
    ipady=3
)


stroke_card = tk.Frame(
    probability_row,
    bg=RED_LIGHT,
    height=90,
    highlightbackground="#F1CDD3",
    highlightthickness=1
)

stroke_card.pack(
    side="left",
    expand=True,
    fill="both",
    padx=(8, 0)
)

stroke_card.pack_propagate(False)


stroke_top = tk.Frame(
    stroke_card,
    bg=RED_LIGHT
)

stroke_top.pack(
    fill="x",
    padx=18,
    pady=(12, 5)
)


tk.Label(
    stroke_top,
    text="STROKE",
    font=(
        "Segoe UI",
        10,
        "bold"
    ),
    fg=RED,
    bg=RED_LIGHT
).pack(
    side="left"
)


tk.Label(
    stroke_top,
    textvariable=stroke_prob_var,
    font=(
        "Segoe UI",
        18,
        "bold"
    ),
    fg=RED,
    bg=RED_LIGHT
).pack(
    side="right"
)


stroke_bar = ttk.Progressbar(
    stroke_card,
    style="Stroke.Horizontal.TProgressbar",
    maximum=100
)

stroke_bar.pack(
    fill="x",
    padx=18,
    pady=(3, 12),
    ipady=3
)

stats_row = tk.Frame(
    result_card,
    bg=WHITE
)

stats_row.pack(
    fill="x",
    padx=35,
    pady=(2, 10)
)




confidence_box = tk.Frame(
    stats_row,
    bg=BLUE_LIGHT,
    height=60
)

confidence_box.pack(
    side="left",
    expand=True,
    fill="both",
    padx=(0, 8)
)

confidence_box.pack_propagate(False)


tk.Label(
    confidence_box,
    text="Prediction Confidence",
    font=(
        "Segoe UI",
        8
    ),
    fg=MUTED,
    bg=BLUE_LIGHT
).pack(
    pady=(8, 0)
)


tk.Label(
    confidence_box,
    textvariable=confidence_var,
    font=(
        "Segoe UI",
        16,
        "bold"
    ),
    fg=BLUE,
    bg=BLUE_LIGHT
).pack()



windows_box = tk.Frame(
    stats_row,
    bg="#F1F4F7",
    height=60
)

windows_box.pack(
    side="left",
    expand=True,
    fill="both",
    padx=(8, 0)
)

windows_box.pack_propagate(False)


tk.Label(
    windows_box,
    text="Analysed Windows",
    font=(
        "Segoe UI",
        8
    ),
    fg=MUTED,
    bg="#F1F4F7"
).pack(
    pady=(8, 0)
)


tk.Label(
    windows_box,
    textvariable=windows_var,
    font=(
        "Segoe UI",
        16,
        "bold"
    ),
    fg=NAVY,
    bg="#F1F4F7"
).pack()


status_bar = tk.Frame(
    root,
    bg=NAVY,
    height=28
)

status_bar.pack(
    fill="x"
)

status_bar.pack_propagate(False)


tk.Label(
    status_bar,
    textvariable=status_var,
    font=(
        "Segoe UI",
        8
    ),
    fg="#DCE9F5",
    bg=NAVY
).pack(
    side="left",
    padx=16,
    pady=5
)


tk.Label(
    status_bar,
    text="CNN-LSTM  •  Bilateral Foot IMU  •  104 Hz  •  12 Channels",
    font=(
        "Segoe UI",
        8
    ),
    fg="#9DBCD7",
    bg=NAVY
).pack(
    side="right",
    padx=16,
    pady=5
)


root.mainloop()