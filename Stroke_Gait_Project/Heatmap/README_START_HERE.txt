NeuroGait Heatmap Prototype — 202 validated images

WHERE TO EXTRACT:
Inside your INNER project folder:
C:\Users\Anam\Desktop\Stroke_Gait_Project\Stroke_Gait_Project\

After extraction, this folder will contain:
  neurogait App_heatmap.py
  heatmap_module.py
  data/heatmaps/valid/ (202 images)
  data/heatmaps/heatmap_inventory.csv
  data/heatmaps/invalid_zero_byte_entries.txt

START APP from OUTER folder (venv already active):
  python -m streamlit run "Stroke_Gait_Project\neurogait App_heatmap.py"

WORKFLOW:
1. Select Stroke or Parkinson's tab.
2. Enter dataset participant ID (not a name) and upload relevant IMU CSV(s).
3. Click Analyse. The real CNN-LSTM prediction appears as before.
4. Below it, a reference plantar heatmap and approximate image-based footprint indicators appear.
5. The same participant ID gets the same reference image. Different IDs receive distinct images until 202 are used.

SCOPE AND LIMITATIONS:
- 400 PNG entries in source archive: 202 valid, 198 zero-byte.
- There are only 202 distinct valid reference images. 239 different references require 37 more valid source files.
- These foot images are NOT linked to the Stroke or Parkinson's dataset participants.
- Region measures are approximate IMAGE PIXEL contact features, NOT measured mm/cm or calibrated kPa.
- Temperature, true 3D arch height and pressure in kPa are not recoverable from colour images alone.
- No patient-specific plantar pressure, disease label or DFU diagnosis is inferred from a reference image.
- Existing Stroke and Parkinson's checkpoints are never modified or retrained.
- If using these in a student demonstration or report, identify them as independent reference scans.
