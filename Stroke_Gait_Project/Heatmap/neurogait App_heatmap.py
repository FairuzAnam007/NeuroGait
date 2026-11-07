from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from heatmap_module import show_reference_heatmap


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PD_SRC = SRC / "parkinson"
for folder in (SRC, PD_SRC):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

STROKE_MODEL = ROOT / "models" / "cnn_lstm_best.pt"
PD_MODEL = ROOT / "models" / "parkinson" / "final_multitask" / "pd_final_cnn_lstm.pt"

st.set_page_config(page_title="NeuroGait | Gait Screening", page_icon="🦶", layout="wide")
st.title("NeuroGait")
st.caption("Bilateral foot IMU gait analysis | Stroke and Parkinson's research screening")



@st.cache_resource(show_spinner="Loading existing model modules...")
def load_prediction_functions():

    from predict import predict_gait
    from preprocess import synchronize_feet
    from pd_final_predict import predict_signal
    from pd_preprocess import synchronize_pd_feet
    return predict_gait, synchronize_feet, predict_signal, synchronize_pd_feet


def result_summary(label, condition_score, condition_name, windows):
    other_score = 100.0 - condition_score
    st.subheader("Screening result")
    st.info(label)
    a, b, c = st.columns(3)
    a.metric(f"{condition_name} model score", f"{condition_score:.2f}%")
    b.metric("Healthy model score", f"{other_score:.2f}%")
    c.metric("Analysed windows", f"{int(windows):,}")
    st.progress(float(np.clip(condition_score / 100.0, 0, 1)))
    st.caption("Scores reflect the trained model's output; they are not calibrated clinical disease probabilities.")


def signal_visualisations(signal, sampling_rate):

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 12 or len(signal) < 2:
        st.warning("IMU visualisations require a 12-channel bilateral signal.")
        return

    n = min(len(signal), int(sampling_rate * 15))
    excerpt = signal[:n]
    t = np.arange(n) / sampling_rate

    left_acc = np.linalg.norm(excerpt[:, :3], axis=1)
    right_acc = np.linalg.norm(excerpt[:, 6:9], axis=1)
    chart_df = pd.DataFrame({
        "Time (s)": t,
        "Left acceleration magnitude": left_acc,
        "Right acceleration magnitude": right_acc,
    }).set_index("Time (s)")

    st.subheader("Bilateral IMU signals")
    st.line_chart(chart_df)
    st.caption("First 15 seconds (or less). Acceleration magnitude uses the source sensor's units.")

    with st.expander("Show 12-channel IMU signal heatmap", expanded=False):
        mu = excerpt.mean(axis=0, keepdims=True)
        sigma = excerpt.std(axis=0, keepdims=True)
        z = (excerpt - mu) / np.maximum(sigma, 1e-6)
        channel_labels = [
            "L Acc X", "L Acc Y", "L Acc Z", "L Gyro X", "L Gyro Y", "L Gyro Z",
            "R Acc X", "R Acc Y", "R Acc Z", "R Gyro X", "R Gyro Y", "R Gyro Z"
        ]
        fig, ax = plt.subplots(figsize=(12, 4.6))
        im = ax.imshow(
            np.clip(z.T, -3, 3), aspect="auto", origin="upper", cmap="coolwarm",
            extent=(0, n / sampling_rate, 11.5, -0.5), vmin=-3, vmax=3
        )
        ax.set_yticks(range(12), channel_labels)
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Standardised IMU channel variation")
        fig.colorbar(im, ax=ax, label="Standardised signal (z-score)", fraction=0.025)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Generated directly from the uploaded IMU recording. This is NOT a plantar-pressure or temperature map.")


def display_model_checkpoints():
    if not STROKE_MODEL.is_file():
        st.caption("Stroke checkpoint not found at models/cnn_lstm_best.pt. Check the project folder before prediction.")
    if not PD_MODEL.is_file():
        st.caption("PD checkpoint not found at models/parkinson/final_multitask/pd_final_cnn_lstm.pt.")


stroke_tab, pd_tab = st.tabs([
    "🧠 Stroke screening", "🚶 Parkinson's screening"
])

with stroke_tab:
    st.subheader("Stroke-associated gait screening")
    st.write("Upload **two Stroke-format CSVs**: one left-foot and one right-foot recording (104 Hz protocol).")
    left = st.file_uploader("Left foot CSV", type=["csv"], key="stroke_left")
    right = st.file_uploader("Right foot CSV", type=["csv"], key="stroke_right")
    stroke_participant = st.text_input("Stroke participant ID (for a fixed reference image)", key="stroke_participant")
    if st.button("Analyse Stroke recording", type="primary", key="stroke_go"):
        if left is None or right is None:
            st.warning("Select both left- and right-foot CSV files.")
        else:
            try:
                predict_gait, synchronize_feet, _, _ = load_prediction_functions()
                with tempfile.TemporaryDirectory() as tmp_dir:
                    left_path = Path(tmp_dir) / "left.csv"
                    right_path = Path(tmp_dir) / "right.csv"
                    left_path.write_bytes(left.getvalue())
                    right_path.write_bytes(right.getvalue())
                    result = predict_gait(left_path, right_path)
                    signal = np.asarray(synchronize_feet(left_path, right_path)).copy()
                st.session_state["stroke_result"] = (result, signal, stroke_participant.strip())
            except Exception as error:
                st.session_state.pop("stroke_result", None)
                st.error(f"Stroke analysis failed: {error}")
    if "stroke_result" in st.session_state:
        result, signal, linked_id = st.session_state["stroke_result"]
        result_summary(result["prediction"], float(result["stroke_probability"]), "Stroke", result["windows"])
        show_reference_heatmap("stroke", linked_id)
        signal_visualisations(signal, 104)

with pd_tab:
    st.subheader("Parkinson's-associated gait screening")
    st.write("Upload **one combined PD-format CSV** containing left and right foot samples (36 Hz protocol).")
    pd_file = st.file_uploader("Combined PD foot recording CSV", type=["csv"], key="pd_file")
    pd_participant = st.text_input("Parkinson’s participant ID (for a fixed reference image)", key="pd_participant")
    if st.button("Analyse Parkinson's recording", type="primary", key="pd_go"):
        if pd_file is None:
            st.warning("Select a combined PD-format CSV file.")
        else:
            try:
                _, _, predict_signal, synchronize_pd_feet = load_prediction_functions()
                df = pd.read_csv(pd_file)
                required = {"foot", "timestamp", "acx", "acy", "acz", "gyrx", "gyry", "gyrz"}
                missing = sorted(required - set(df.columns))
                if missing:
                    raise ValueError("Missing CSV columns: " + ", ".join(missing))
                result = predict_signal(df)
                signal = np.asarray(synchronize_pd_feet(df)).copy()
                st.session_state["pd_result"] = (result, signal, pd_participant.strip())
            except Exception as error:
                st.session_state.pop("pd_result", None)
                st.error(f"Parkinson's analysis failed: {error}")
    if "pd_result" in st.session_state:
        result, signal, linked_id = st.session_state["pd_result"]
        result_summary(result["prediction"], 100 * float(result["pd_probability"]), "Parkinson's", result["windows"])
        show_reference_heatmap("pd", linked_id)
        signal_visualisations(signal, 36)

st.divider()
display_model_checkpoints()
st.caption("The Stroke and PD datasets have different input formats and sensor sampling rates. Do not upload an arbitrary file to both models as if their inputs were interchangeable.")
