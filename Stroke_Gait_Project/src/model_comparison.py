import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


data = {
    "Model": [
        "1D CNN",
        "LSTM",
        "CNN-LSTM"
    ],

    "Accuracy (%)": [
        93.03,
        93.82,
        94.41
    ],

    "Precision (%)": [
        96.51,
        98.19,
        96.46
    ],

    "Recall (%)": [
        95.44,
        94.65,
        97.15
    ],

    "F1-score (%)": [
        95.97,
        96.39,
        96.80
    ]
}


df = pd.DataFrame(data)


print("MODEL COMPARISON")
print("--------------------------------")

print(
    df.to_string(index=False)
)


df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)

plot_df = df.set_index("Model")

plot_df.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.ylabel("Score (%)")
plt.xlabel("Model")

plt.title(
    "Comparison of Deep Learning Models"
)

plt.ylim(
    90,
    100
)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Metric"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


save_path = (
    RESULT_DIR
    / "model_comparison.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)


print("\nGraph saved to:")
print(save_path)


print("\nBest Model: CNN-LSTM")

print(
    "Reason: Lowest validation loss "
    "and strongest overall test performance."
)


plt.show()

