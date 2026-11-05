# Crop Map Generator (RF / ML / DL)

## Setup in VS Code
1. Install Python 3.10+ and the VS Code *Python* extension. Open the `CropMapper` folder.
2. Terminal -> `python -m venv .venv` then `.venv\Scripts\activate`
3. Rasterio/GeoPandas wheels install fine on Windows with pip (Python 3.10-3.12):
   `pip install -r requirements.txt`   (PyTorch CPU/GPU: see pytorch.org if `torch` fails)
4. Check the two shapefile paths in `config.py` (already set to your D:\NEW_MH_District files).
   If auto-detection of the name columns picks the wrong field, set `DISTRICT_NAME_FIELD` / `TALUKA_NAME_FIELD`.
5. Press F5 (or `python app.py`).

## Workflow
1. **Add image(s)** - one multi-band FCC/NDVI stack or several files (different grids are resampled to the first).
2. Choose **District** (+ optional **Taluka**).
3. **Open GT shapefile** (points or polygons), pick the **class label column**.
4. **Clip GT...** - GT is clipped to the area; features on 0/NoData pixels or outside the image are
   printed (ID + reason) in the Log and removed. Map + class-wise counts appear in the tabs.
5. Set Train % / Test %, tick up to **3 methods**, choose output folder, **RUN**.

## Outputs (output folder)
* `<District>_<Taluka>_<RF|XGB|...>.tif` - classified map (uint8, 0 = NoData/outside, 1..N classes, N+1 = Other crop, colour table embedded) + `_legend.csv`
* `<District>_<Taluka>_accuracy_assessment.xlsx` - Summary, Training_counts, Removed_GT and one sheet per method with
  Overall Accuracy, Kappa, error matrix, Producer's & User's accuracy
* In the GUI *Results* tab: classified map, optional test-point overlay (green correct / red wrong), error matrix, PA/UA.

## Notes
* **Other crop**: valid pixels whose best-class probability is below "Min confidence" (default 0.40) are put in *Other crop*; also counted in the error matrix as a class. Set 0 to disable. (SVM gives no probability, so it never assigns Other.)
* Train/test split is done per class; with "split by polygon/point" on, whole polygons go to either train or test (no leakage).
* Pixel models (RF, XGB, SVM, DT, MLC, CNN, LSTM, Transformer) use each pixel's band vector; LSTM/Transformer treat
  multiple input files as time steps (single file: bands as the sequence). U-Net trains on 32x32 patches around GT pixels.
* Whole district at 10 m can need several GB RAM; start with a taluka. SVM prediction over a large area is slow.
