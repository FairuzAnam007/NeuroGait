from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# 1. PROJECT PATHS: works with the GUI inside the Heatmap folder
# ------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

# The innermost project directory has src/, models/, and data/.
# Search upward to handle extra Stroke_Gait_Project nesting.
ROOT = next(
    (
        candidate for candidate in (APP_DIR, *APP_DIR.parents)
        if (candidate / "src" / "predict.py").is_file()
        and (candidate / "models").is_dir()
    ),
    None,
)

if ROOT is None:
    raise FileNotFoundError(
        "Cannot find the inner project folder containing "
        "src/predict.py and models/. Check the copied project structure."
    )

SRC = ROOT / "src"
PD_SRC = SRC / "parkinson"

for folder in (APP_DIR, SRC, PD_SRC):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

# Load only this GUI's adjacent heatmap module, not any duplicate copy
# from the parent folder.
from heatmap_module import show_heatmap_panel

STROKE_MODEL = ROOT / "models" / "cnn_lstm_best.pt"
STROKE_MEAN = ROOT / "models" / "channel_mean.npy"
STROKE_STD = ROOT / "models" / "channel_std.npy"

PD_MODEL = (
    ROOT / "models" / "parkinson"
    / "final_multitask" / "pd_final_cnn_lstm.pt"
)
PD_MEAN = PD_MODEL.parent / "pd_final_mean.npy"
PD_STD = PD_MODEL.parent / "pd_final_std.npy"


# ------------------------------------------------------------
# 2. PAGE CONFIGURATION AND PREMIUM DESIGN
# ------------------------------------------------------------

st.set_page_config(
    page_title="NeuroGait | Intelligent Gait Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            ellipse at 12% 0%,
            rgba(24,104,137,.20),
            transparent 38%
        ),
        radial-gradient(
            ellipse at 96% 8%,
            rgba(10,145,141,.10),
            transparent 35%
        ),
        #0a1120;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

/* NeuroGait main banner */

.neurogait-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        120deg,
        #162a43 0%,
        #121b32 58%,
        #102d37 100%
    );
    border: 1px solid rgba(108,217,224,.25);
    border-radius: 23px;
    padding: 30px 38px 28px;
    margin-bottom: 22px;
    box-shadow: 0 18px 46px rgba(0,0,0,.20);
}

.neurogait-hero:after {
    content: "";
    position: absolute;
    right: -60px;
    top: -95px;
    width: 270px;
    height: 270px;
    border-radius: 50%;
    background: rgba(39,228,202,.085);
    filter: blur(18px);
    pointer-events: none;
}

.ng-heading {
    display: flex;
    align-items: center;
    gap: 19px;
}

.ng-logo {
    width: 66px;
    height: 66px;
    flex-shrink: 0;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(145deg,#174b65,#193a52);
    border: 1px solid rgba(94,224,211,.37);
    box-shadow: 0 0 25px rgba(25,204,189,.10);
    color: #78f2e4;
    font: 800 22px 'Segoe UI',sans-serif;
}

.ng-title {
    margin: 0 !important;
    padding: 0 !important;
    font: 800 clamp(38px,5vw,59px)/1.12 'Segoe UI',sans-serif !important;
    letter-spacing: -2px;
    color: #f5f9ff;
}

.ng-title span {
    background: linear-gradient(90deg,#68efdf,#4fb9ef);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.ng-eyebrow {
    margin-top: 7px;
    font: 700 11px 'Segoe UI',sans-serif;
    letter-spacing: 2.2px;
    text-transform: uppercase;
    color: #9cd7dc;
}

.ng-description {
    margin: 22px 0 15px;
    color: #c4d4e7;
    font-size: 15px;
}

.ng-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.ng-tag {
    padding: 6px 13px;
    border-radius: 30px;
    background: rgba(75,208,204,.085);
    border: 1px solid rgba(95,205,212,.24);
    color: #bbefec;
    font-size: 12px;
}

.ng-note {
    margin-top: 15px;
    color: #a6b9c8;
    font-size: 12px;
}

/* Navigation tabs */

button[data-baseweb="tab"] {
    font-weight: 650;
    padding: 11px 19px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #68e9df !important;
}

div[data-baseweb="tab-highlight"] {
    background: #4ce0d2 !important;
}

/* Result cards */

[data-testid="stMetric"] {
    background: linear-gradient(135deg,#172b43,#101e31);
    border: 1px solid rgba(110,173,196,.20);
    border-radius: 16px;
    padding: 17px 20px;
}

[data-testid="stMetricLabel"] {
    color: #b8cede;
}

[data-testid="stMetricValue"] {
    color: #f5fcff;
}

/* File upload boxes */

[data-testid="stFileUploader"] {
    background: rgba(25,44,65,.42);
    border: 1px solid rgba(111,172,197,.20);
    border-radius: 14px;
    padding: 9px 12px;
}

/* Analyse buttons */

div.stButton > button[kind="primary"] {
    background: linear-gradient(110deg,#0ea7a7,#137fb1);
    color: #fff;
    border: 0;
    border-radius: 12px;
    font-weight: 700;
    min-height: 46px;
    padding: 10px 24px;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(110deg,#13c4b6,#158cbe);
    color: white;
    border: 0;
}

/* Images and tables */

[data-testid="stImage"] img {
    border-radius: 14px;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: rgba(117,160,181,.18);
}

h2, h3 {
    letter-spacing: -.3px;
}

/* Mobile layout */

@media(max-width:700px) {
    .neurogait-hero {
        padding: 24px 18px;
    }

    .ng-logo {
        width: 49px;
        height: 49px;
        border-radius: 14px;
        font-size: 17px;
    }

    .ng-heading {
        gap: 13px;
    }

    .ng-eyebrow {
        letter-spacing: 1.1px;
    }
}

/* =========================================
   FIX DARK AND UNREADABLE TEXT
   ========================================= */

/* All main headings */
.block-container h1,
.block-container h2,
.block-container h3,
.block-container h4,
.block-container h5 {
    color: #F5F9FF !important;
}

/* Paragraphs and descriptions */
.block-container [data-testid="stMarkdownContainer"] p,
.block-container [data-testid="stMarkdownContainer"] li {
    color: #E1EBFA !important;
}

/* Upload labels and input labels */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stFileUploader"] label,
[data-testid="stTextInput"] label {
    color: #F0F6FF !important;
    opacity: 1 !important;
}

/* Stroke and Parkinson's tabs */
button[data-baseweb="tab"],
button[data-baseweb="tab"] p {
    color: #D7E5F5 !important;
    opacity: 1 !important;
    font-weight: 650 !important;
}

/* Selected tab */
button[data-baseweb="tab"][aria-selected="true"],
button[data-baseweb="tab"][aria-selected="true"] p {
    color: #62F0DE !important;
}

/* File uploader: dark background */
[data-testid="stFileUploaderDropzone"] {
    background: #17283E !important;
    border: 1px dashed #49718D !important;
    border-radius: 12px !important;
}

/* File uploader instructions */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzone"] small {
    color: #DAE8F8 !important;
    opacity: 1 !important;
}

/* Browse button */
[data-testid="stFileUploaderDropzone"] button {
    background: #244763 !important;
    color: #FFFFFF !important;
    border: 1px solid #456B86 !important;
}

/* Text inputs */
[data-testid="stTextInput"] input {
    background: #17283E !important;
    color: #FFFFFF !important;
    border: 1px solid #456B86 !important;
}

/* Metric labels and values */
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"] * {
    color: #F1F7FF !important;
    opacity: 1 !important;
}

/* Captions */
[data-testid="stCaptionContainer"] p {
    color: #BBCDDD !important;
}

/* Keep the NeuroGait gradient branding */
.ng-title span {
    background: linear-gradient(90deg, #6AF4E3, #55C9FF);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------
# 3. FIXED NEUROGAIT HEADER
# ------------------------------------------------------------

# Keep HTML on one line to prevent Markdown
# from displaying indented HTML as a code block.

HERO_HTML = (
    '<div class="neurogait-hero">'
    '<div class="ng-heading">'
    '<div class="ng-logo">NG</div>'
    '<div>'
    '<h1 class="ng-title">Neuro<span>Gait</span></h1>'
    '<div class="ng-eyebrow">'
    'Intelligent Gait Screening System'
    '</div>'
    '</div>'
    '</div>'
    '<div class="ng-description">'
    "Deep Learning-Based Stroke and Parkinson's Gait "
    'Analysis Using Bilateral Foot IMU Signals'
    '</div>'
    '<div class="ng-tags">'
    '<span class="ng-tag">CNN-LSTM Analysis</span>'
    '<span class="ng-tag">Bilateral Foot IMU</span>'
    '<span class="ng-tag">Plantar Heatmap</span>'
    '<span class="ng-tag">Foot-Contact Analysis</span>'
    '</div>'
    '<div class="ng-note">'
    'Academic screening demonstration — not a clinical diagnosis'
    '</div>'
    '</div>'
)

st.markdown(
    HERO_HTML,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# 4. LOAD EXISTING TRAINED MODELS
# ------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Stroke inference module...")
def load_stroke_functions():
    from predict import predict_gait
    from preprocess import synchronize_feet
    return predict_gait, synchronize_feet


@st.cache_resource(show_spinner="Loading Parkinson's inference module...")
def load_pd_functions():
    from pd_final_predict import predict_signal
    from pd_preprocess import synchronize_pd_feet
    return predict_signal, synchronize_pd_feet


# ------------------------------------------------------------
# 5. SCREENING RESULT DISPLAY
# ------------------------------------------------------------

def result_summary(
    label,
    condition_score,
    condition_name,
    windows
):

    condition_score = float(
        np.clip(condition_score, 0, 100)
    )

    healthy_score = 100.0 - condition_score

    st.subheader("Screening Result")

    st.info(label)

    a, b, c = st.columns(3)

    a.metric(
        f"{condition_name} Model Score",
        f"{condition_score:.2f}%"
    )

    b.metric(
        "Healthy Model Score",
        f"{healthy_score:.2f}%"
    )

    c.metric(
        "Analysed Windows",
        f"{int(windows):,}"
    )

    st.progress(
        condition_score / 100.0
    )

    st.caption(
        "These are trained-model scores, not "
        "calibrated clinical disease probabilities."
    )


# ------------------------------------------------------------
# 6. IMU SIGNAL VISUALISATION
# ------------------------------------------------------------

def signal_visualisations(
    signal,
    sampling_rate
):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    if (
        signal.ndim != 2
        or signal.shape[1] != 12
        or len(signal) < 2
    ):

        st.warning(
            "IMU charts need a bilateral "
            "12-channel signal."
        )

        return

    n = min(
        len(signal),
        int(sampling_rate * 15)
    )

    excerpt = signal[:n]

    time_seconds = np.arange(n) / sampling_rate

    left_acc = np.linalg.norm(
        excerpt[:, 0:3],
        axis=1
    )

    right_acc = np.linalg.norm(
        excerpt[:, 6:9],
        axis=1
    )

    st.subheader("Bilateral IMU Signals")

    chart = pd.DataFrame({
        "Time (s)": time_seconds,
        "Left acceleration magnitude": left_acc,
        "Right acceleration magnitude": right_acc
    }).set_index("Time (s)")

    st.line_chart(chart)

    st.caption(
        "First 15 seconds of the recording; "
        "original sensor units are retained."
    )

    with st.expander(
        "Show 12-Channel IMU Signal Heatmap",
        expanded=False
    ):

        avg = excerpt.mean(
            axis=0,
            keepdims=True
        )

        std = excerpt.std(
            axis=0,
            keepdims=True
        )

        standardised = (
            excerpt - avg
        ) / np.maximum(std, 1e-6)

        labels = [
            "L Acc X",
            "L Acc Y",
            "L Acc Z",
            "L Gyro X",
            "L Gyro Y",
            "L Gyro Z",
            "R Acc X",
            "R Acc Y",
            "R Acc Z",
            "R Gyro X",
            "R Gyro Y",
            "R Gyro Z"
        ]

        fig, ax = plt.subplots(
            figsize=(12, 4.6)
        )

        img = ax.imshow(
            np.clip(standardised.T, -3, 3),
            aspect="auto",
            origin="upper",
            cmap="coolwarm",
            extent=(
                0,
                n / sampling_rate,
                11.5,
                -0.5
            ),
            vmin=-3,
            vmax=3
        )

        ax.set_yticks(
            range(12),
            labels
        )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_title(
            "Standardised IMU Channel Variation"
        )

        fig.colorbar(
            img,
            ax=ax,
            label="Standardised signal (z-score)",
            fraction=.025
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# ------------------------------------------------------------
# 7. MODEL FILE CHECK
# ------------------------------------------------------------

def model_checkpoint_notes():
    with st.expander("Project file paths and setup diagnostics", expanded=False):
        paths = {
            "GUI": Path(__file__).resolve(),
            "Inner project": ROOT,
            "Stroke predictor": SRC / "predict.py",
            "Stroke checkpoint": STROKE_MODEL,
            "Stroke mean": STROKE_MEAN,
            "Stroke std": STROKE_STD,
            "PD predictor": PD_SRC / "pd_final_predict.py",
            "PD checkpoint": PD_MODEL,
            "PD mean": PD_MEAN,
            "PD std": PD_STD,
            "Heatmap module": APP_DIR / "heatmap_module.py",
        }
        for title, path in paths.items():
            st.write(f"{'✅' if path.exists() else '❌'} **{title}:** `{path}`")

    stroke_missing = [p for p in (STROKE_MODEL, STROKE_MEAN, STROKE_STD) if not p.is_file()]
    pd_missing = [p for p in (PD_MODEL, PD_MEAN, PD_STD) if not p.is_file()]
    if stroke_missing:
        st.warning("Stroke model files missing: " + ", ".join(str(p) for p in stroke_missing))
    if pd_missing:
        st.warning("PD model files missing: " + ", ".join(str(p) for p in pd_missing))


def remove_saved_result(key):

    st.session_state.pop(
        key,
        None
    )


# ------------------------------------------------------------
# 8. MAIN SCREENING TABS
# ------------------------------------------------------------

stroke_tab, pd_tab = st.tabs([
    "🧠 Stroke Screening",
    "🚶 Parkinson's Screening"
])


# ------------------------------------------------------------
# 9. STROKE SCREENING
# ------------------------------------------------------------

with stroke_tab:

    st.subheader(
        "Stroke-Associated Gait Screening"
    )

    st.write(
        "Upload separate left- and right-foot "
        "Stroke IMU CSVs (104 Hz protocol)."
    )

    left_csv = st.file_uploader(
        "Left Foot CSV",
        type=["csv"],
        key="stroke_left",
        on_change=remove_saved_result,
        args=("stroke_result",)
    )

    right_csv = st.file_uploader(
        "Right Foot CSV",
        type=["csv"],
        key="stroke_right",
        on_change=remove_saved_result,
        args=("stroke_result",)
    )

    stroke_id = st.text_input(
        "Stroke Participant ID",
        key="stroke_participant"
    )

    if st.button(
        "Analyse Stroke Recording",
        type="primary",
        key="stroke_go"
    ):

        if left_csv is None or right_csv is None:

            st.warning(
                "Upload both left- and right-foot CSVs."
            )

        else:

            try:

                predict_gait, synchronize_feet = load_stroke_functions()

                with tempfile.TemporaryDirectory() as tmp_dir:

                    left_path = (
                        Path(tmp_dir) / "left.csv"
                    )

                    right_path = (
                        Path(tmp_dir) / "right.csv"
                    )

                    left_path.write_bytes(
                        left_csv.getvalue()
                    )

                    right_path.write_bytes(
                        right_csv.getvalue()
                    )

                    with st.spinner(
                        "Analysing Stroke IMU recording..."
                    ):

                        result = predict_gait(
                            left_path,
                            right_path
                        )

                        signal = np.asarray(
                            synchronize_feet(
                                left_path,
                                right_path
                            )
                        ).copy()

                st.session_state["stroke_result"] = (
                    result,
                    signal,
                    stroke_id.strip()
                )

            except Exception as error:

                st.session_state.pop(
                    "stroke_result",
                    None
                )

                st.error(
                    f"Stroke analysis failed: {error}"
                )

    # Stroke result
    if "stroke_result" in st.session_state:

        result, signal, linked_id = (
            st.session_state["stroke_result"]
        )

        result_summary(
            result["prediction"],
            float(result["stroke_probability"]),
            "Stroke",
            result["windows"]
        )

        st.divider()

        # Plantar heatmap
        show_heatmap_panel(
            "stroke",
            linked_id
        )

        st.divider()

        # Bilateral IMU
        signal_visualisations(
            signal,
            104
        )


# ------------------------------------------------------------
# 10. PARKINSON'S SCREENING
# ------------------------------------------------------------

with pd_tab:

    st.subheader(
        "Parkinson's-Associated Gait Screening"
    )

    st.write(
        "Upload one combined left/right-foot "
        "PD IMU CSV (36 Hz processed protocol)."
    )

    pd_file = st.file_uploader(
        "Combined PD Foot Recording CSV",
        type=["csv"],
        key="pd_file",
        on_change=remove_saved_result,
        args=("pd_result",)
    )

    pd_id = st.text_input(
        "Parkinson's Participant ID",
        key="pd_participant"
    )

    if st.button(
        "Analyse Parkinson's Recording",
        type="primary",
        key="pd_go"
    ):

        if pd_file is None:

            st.warning(
                "Upload a combined Parkinson's CSV."
            )

        else:

            try:

                predict_signal, synchronize_pd_feet = load_pd_functions()

                frame = pd.read_csv(
                    pd_file
                )

                required = {
                    "foot",
                    "timestamp",
                    "acx",
                    "acy",
                    "acz",
                    "gyrx",
                    "gyry",
                    "gyrz"
                }

                missing = sorted(
                    required - set(frame.columns)
                )

                if missing:

                    raise ValueError(
                        "Missing CSV columns: "
                        + ", ".join(missing)
                    )

                with st.spinner(
                    "Analysing Parkinson's IMU recording..."
                ):

                    result = predict_signal(
                        frame
                    )

                    signal = np.asarray(
                        synchronize_pd_feet(
                            frame
                        )
                    ).copy()

                st.session_state["pd_result"] = (
                    result,
                    signal,
                    pd_id.strip()
                )

            except Exception as error:

                st.session_state.pop(
                    "pd_result",
                    None
                )

                st.error(
                    f"Parkinson's analysis failed: {error}"
                )

    # Parkinson's result
    if "pd_result" in st.session_state:

        result, signal, linked_id = (
            st.session_state["pd_result"]
        )

        result_summary(
            result["prediction"],
            100.0 * float(
                result["pd_probability"]
            ),
            "Parkinson's",
            result["windows"]
        )

        st.divider()

        # Plantar heatmap
        show_heatmap_panel(
            "pd",
            linked_id
        )

        st.divider()

        # Bilateral IMU
        signal_visualisations(
            signal,
            36
        )


# ------------------------------------------------------------
# 11. FOOTER
# ------------------------------------------------------------

st.divider()

model_checkpoint_notes()

st.caption(
    "Stroke and Parkinson's input formats "
    "and sampling rates are different."
)
