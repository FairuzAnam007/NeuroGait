
"""Plantar heatmap display and foot-contact measurements."""

from pathlib import Path
import csv

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

HEATMAP_DIR = (
    PROJECT_ROOT / "data" / "heatmaps" / "valid"
)

# Existing assignments preserved
ASSIGNMENTS_FILE = (
    PROJECT_ROOT / "data" / "heatmaps"
    / "reference_assignments.csv"
)


# =========================================
# 1. ASSIGN ONE IMAGE PER PARTICIPANT
# =========================================

def assign_heatmap(participant_key):

    images = sorted(
        p for p in HEATMAP_DIR.glob("*.png")
        if p.stat().st_size > 0
    )

    if not images:
        return None, (
            "No heatmap images found. "
            "Check data/heatmaps/valid/"
        )

    assignments = {}

    if ASSIGNMENTS_FILE.is_file():

        with ASSIGNMENTS_FILE.open(
            "r", newline="", encoding="utf-8"
        ) as file:

            for row in csv.DictReader(file):
                assignments[row["participant_key"]] = (
                    row["image_file"]
                )

    # Return previously assigned image
    if participant_key in assignments:

        path = (
            HEATMAP_DIR / assignments[participant_key]
        )

        if path.is_file() and path.stat().st_size > 0:
            return path, None

        return None, (
            f"Heatmap file missing for "
            f"participant {participant_key}."
        )

    # Choose an unused image
    occupied = set(assignments.values())

    unused = [
        path for path in images
        if path.name not in occupied
    ]

    if not unused:
        return None, (
            "No unused heatmap images remain. "
            "Add more valid images."
        )

    selected = unused[0]

    ASSIGNMENTS_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with ASSIGNMENTS_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as file:

        write_header = file.tell() == 0

        writer = csv.writer(file)

        if write_header:
            writer.writerow([
                "participant_key",
                "image_file"
            ])

        writer.writerow([
            participant_key,
            selected.name
        ])

    return selected, None


# =========================================
# 2. REMOVE EXTRA BLACK MARGINS
# =========================================

def crop_black_margin(image, padding_ratio=0.07):

    rgb = np.asarray(image.convert("RGB"))

    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)

    coloured = (
        (mx >= 50) &
        ((mx - mn) >= 30)
    )

    rows = np.flatnonzero(
        coloured.sum(axis=1) >= 5
    )

    cols = np.flatnonzero(
        coloured.sum(axis=0) >= 5
    )

    if not len(rows) or not len(cols):
        return image

    left = int(cols[0])
    right = int(cols[-1]) + 1

    top = int(rows[0])
    bottom = int(rows[-1]) + 1

    pad = int(
        max(
            right - left,
            bottom - top
        ) * padding_ratio
    )

    left = max(0, left - pad)
    top = max(0, top - pad)

    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)

    return image.crop((
        left, top, right, bottom
    ))


# =========================================
# 3. FOOT CONTACT FEATURE EXTRACTION
# =========================================

def _part_features(rgb, x_start, x_end):

    rgb = rgb[:, x_start:x_end, :]

    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)

    mask = (
        (mx >= 60) &
        ((mx - mn) >= 45)
    )

    column_counts = mask.sum(axis=0)

    active = np.flatnonzero(
        column_counts >= 3
    )

    if len(active) == 0:
        return None

    breaks = (
        np.flatnonzero(np.diff(active) > 4) + 1
    )

    groups = np.split(active, breaks)

    main = max(
        groups,
        key=lambda group: int(
            column_counts[group].sum()
        )
    )

    mask[:, :int(main[0])] = False
    mask[:, int(main[-1]) + 1:] = False

    yy, xx = np.where(mask)

    if len(yy) < 120:
        return None

    y0 = int(yy.min())
    y1 = int(yy.max())

    x0 = int(xx.min())
    x1 = int(xx.max())

    height = max(1, y1 - y0 + 1)
    width = max(1, x1 - x0 + 1)

    crop = mask[
        y0:y1 + 1,
        x0:x1 + 1
    ]

    # Approximate midfoot contact proportion
    ylo = int(round(0.19 * height))

    middle_lo = int(round(0.43 * height))
    middle_hi = int(round(0.68 * height))

    denominator = int(crop[ylo:].sum())

    midcount = int(
        crop[middle_lo:middle_hi].sum()
    )

    midshare = (
        midcount / denominator
        if denominator else 0.0
    )

    def median_width(a, b):

        values = []

        start = int(a * height)
        end = int(b * height)

        for row in crop[start:end]:

            xs = np.flatnonzero(row)

            if len(xs) >= 2:
                values.append(
                    xs[-1] - xs[0] + 1
                )

        if values:
            return float(np.median(values))

        return float("nan")

    return {
        "contact_pixels": int(mask.sum()),
        "length_px": height,
        "width_px": width,
        "forefoot_width_px": median_width(
            0.19, 0.44
        ),
        "heel_width_px": median_width(
            0.77, 0.96
        ),
        "midfoot_contact_share": midshare
    }


# =========================================
# 4. LEFT AND RIGHT FOOT ANALYSIS
# =========================================

def estimate_image_indicators(image):

    rgb = np.asarray(
        image.convert("RGB")
    )

    width = rgb.shape[1]

    left = _part_features(
        rgb,
        int(width * 0.02),
        int(width * 0.48)
    )

    right = _part_features(
        rgb,
        int(width * 0.52),
        int(width * 0.98)
    )

    if not left or not right:
        return None

    la = left["contact_pixels"]
    ra = right["contact_pixels"]

    difference = (
        100 * abs(la - ra) /
        max(1, la + ra)
    )

    return left, right, difference


# =========================================
# 5. GUI HEATMAP PANEL
# =========================================

def show_heatmap_panel(condition, participant_id):

    st.subheader(
        "🦶 Plantar Heatmap & Foot Morphology"
    )

    participant_id = str(
        participant_id or ""
    ).strip()

    if not participant_id:
        st.info(
            "Enter a participant ID above "
            "to display a consistent heatmap."
        )
        return

    path, message = assign_heatmap(
        f"{condition}:{participant_id}"
    )

    if message:
        st.warning(message)
        return

    try:

        with Image.open(path) as original:
            image = original.convert("RGB")

    except (OSError, ValueError):

        st.error(
            "Unable to read the selected heatmap."
        )
        return

    # Improve image presentation
    st.markdown(
        """
        <style>
        div[data-testid="stImage"] img {
            border-radius: 14px;
            border: 1px solid rgba(100,140,175,0.28);
            background-color: #080c12;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col_image, col_metrics = st.columns(
        [1.28, 1],
        gap="large"
    )

    # LEFT PANEL: HEATMAP
    with col_image:

        st.markdown(
            "**Left & Right Plantar Heatmap**"
        )

        clean_image = crop_black_margin(
            image
        )

        st.image(
            clean_image,
            width="stretch"
        )

        st.caption(
            f"Heatmap image: {path.name}"
        )

    # RIGHT PANEL: MEASUREMENTS
    with col_metrics:

        st.markdown(
            "**Foot Morphology Measurements**"
        )

        features = estimate_image_indicators(
            image
        )

        if features is None:

            st.warning(
                "Foot-contact boundaries could "
                "not be estimated for this image."
            )

        else:

            left, right, asymmetry = features

            a, b = st.columns(2)

            a.metric(
                "Left midfoot contact",
                f"{left['midfoot_contact_share'] * 100:.1f}%"
            )

            b.metric(
                "Right midfoot contact",
                f"{right['midfoot_contact_share'] * 100:.1f}%"
            )

            st.metric(
                "Left–right contact difference",
                f"{asymmetry:.1f}%"
            )

            table = pd.DataFrame([
                {
                    "Measurement": "Footprint length (px)",
                    "Left": left["length_px"],
                    "Right": right["length_px"]
                },
                {
                    "Measurement": "Forefoot width (px)",
                    "Left": round(
                        left["forefoot_width_px"], 1
                    ),
                    "Right": round(
                        right["forefoot_width_px"], 1
                    )
                },
                {
                    "Measurement": "Heel width (px)",
                    "Left": round(
                        left["heel_width_px"], 1
                    ),
                    "Right": round(
                        right["heel_width_px"], 1
                    )
                },
                {
                    "Measurement": "Contact area (pixels)",
                    "Left": left["contact_pixels"],
                    "Right": right["contact_pixels"]
                }
            ])

            st.dataframe(
                table,
                hide_index=True,
                width="stretch"
            )

    st.caption(
        "Image source: separate plantar-image dataset. "
        "Measurements are pixel-based estimates."
    )


# Compatibility with earlier GUI imports
show_reference_heatmap = show_heatmap_panel
