from pathlib import Path

import numpy as np
import torch

from models import CNNLSTM
from preprocess import synchronize_feet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "cnn_lstm_best.pt"
MEAN_PATH = PROJECT_ROOT / "models" / "channel_mean.npy"
STD_PATH = PROJECT_ROOT / "models" / "channel_std.npy"


SAMPLING_RATE = 104
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS
STEP_SIZE = WINDOW_SIZE // 2


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = CNNLSTM().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


channel_mean = np.load(MEAN_PATH)
channel_std = np.load(STD_PATH)


def create_windows(data):

    windows = []

    for start in range(
        0,
        len(data) - WINDOW_SIZE + 1,
        STEP_SIZE
    ):

        end = start + WINDOW_SIZE

        windows.append(
            data[start:end]
        )

    return np.array(
        windows,
        dtype=np.float32
    )


def predict_gait(left_file, right_file):


    combined = synchronize_feet(
        left_file,
        right_file
    )

    print(
        "Synchronized shape:",
        combined.shape
    )


    windows = create_windows(
        combined
    )

    if len(windows) == 0:

        raise ValueError(
            "Recording is too short. "
            "At least 5 seconds of synchronized data is required."
        )


    print(
        "Number of windows:",
        len(windows)
    )

    windows = (
        windows - channel_mean
    ) / channel_std


    X = torch.tensor(
        windows,
        dtype=torch.float32
    ).to(device)


    probabilities = []


    with torch.no_grad():

        for start in range(
            0,
            len(X),
            32
        ):

            batch = X[
                start:start + 32
            ]

            outputs = model(
                batch
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )

            probabilities.extend(
                probs[:, 1]
                .cpu()
                .numpy()
            )


    probabilities = np.array(
        probabilities
    )


    stroke_probability = float(
        probabilities.mean()
    )

    healthy_probability = (
        1 - stroke_probability
    )


    if stroke_probability >= 0.50:

        predicted_class = (
            "Stroke-associated gait"
        )

        confidence = (
            stroke_probability
        )

    else:

        predicted_class = (
            "Healthy gait"
        )

        confidence = (
            healthy_probability
        )


    return {
        "prediction": predicted_class,

        "healthy_probability":
            healthy_probability * 100,

        "stroke_probability":
            stroke_probability * 100,

        "confidence":
            confidence * 100,

        "windows":
            len(windows)
    }