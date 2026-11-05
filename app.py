import streamlit as st
import os
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio

# Import your existing backend logic
import core
import models
import unsup
import config

# --- Page Config ---
st.set_page_config(page_title="Crop Map Generator", layout="wide", page_icon="🌾")

# --- Custom CSS for aesthetics ---
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1b5e20; font-weight: bold; margin-bottom: 0; }
    .sub-header { font-size: 1rem; color: #00796b; margin-top: 0; margin-bottom: 2rem; }
    .stButton>button { background-color: #2e7d32; color: white; font-weight: bold; width: 100%; }
    .stButton>button:hover { background-color: #1b5e20; color: white; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🌾 Crop Map Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Developed by ASHISH SELOKAR | Supervised & Unsupervised Classification</p>', unsafe_allow_html=True)

# --- Caching ---
@st.cache_resource
def get_boundaries():
    try:
        return core.load_boundaries()
    except Exception as e:
        st.error(f"Error loading boundaries from `config.py`: {e}")
        return None

bounds = get_boundaries()

if bounds:
    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("1. Input Image(s)")
        uploaded_files = st.file_uploader("Upload FCC/NDVI stack", type=["tif", "tiff", "img"], accept_multiple_files=True)
        
        st.header("2. Study Area")
        districts = core.district_names(bounds)
        sel_district = st.selectbox("Select District", districts)
        
        talukas = ["(whole district)"] + core.taluka_names(bounds, sel_district)
        sel_taluka = st.selectbox("Select Taluka (optional)", talukas)
        taluka_val = None if sel_taluka == "(whole district)" else sel_taluka

        st.header("3. Ground Truth (GT)")
        uploaded_gt = st.file_uploader("Upload GT Shapefile (.zip of .shp+.shx+.dbf+.prj)", type=["zip"])
        
        st.header("4. Sampling & Models")
        train_pct = st.slider("Training %", 10, 90, 70)
        test_pct = 100 - train_pct
        st.write(f"Testing %: {test_pct}")
        
        selected_methods = st.multiselect(
            "Select up to 3 Supervised Methods", 
            options=list(models.METHODS.keys()),
            format_func=lambda x: models.METHODS[x],
            max_selections=3,
            default=["RF"]
        )
        
        st.subheader("Unsupervised Methods")
        run_kmeans = st.checkbox("K-Means", value=False)
        run_isodata = st.checkbox("ISODATA", value=False)
        
        run_btn = st.button("🚀 Run Classification")

    # --- Main Layout ---
    if run_btn:
        if not uploaded_files:
            st.error("Please upload at least one image.")
            st.stop()
        if not uploaded_gt:
            st.error("Ground truth is required for supervised methods.")
            st.stop()

        # Save uploaded files to temp dir (Streamlit Cloud requirement)
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        for f in uploaded_files:
            p = os.path.join(temp_dir, f.name)
            with open(p, "wb") as fh:
                fh.write(f.getbuffer())
            image_paths.append(p)
            
        gt_path = os.path.join(temp_dir, uploaded_gt.name)
        with open(gt_path, "wb") as fh:
            fh.write(uploaded_gt.getbuffer())

        # Logging area
        log_area = st.empty()
        logs = []
        def log_msg(msg):
            logs.append(msg)
            log_area.text_area("Log Output", value="\n".join(logs), height=200, disabled=True)

        with st.spinner("Processing data and running models..."):
            try:
                # 1. Load Region
                region = core.select_region(bounds, sel_district, taluka_val)
                
                # 2. Load Stack
                img = core.load_stack(image_paths, region, log_msg)
                
                # 3. Prepare GT
                gt = core.load_gt(gt_path)
                label_col = [c for c in gt.columns if c != 'geometry'][0] # Auto-pick first column
                prep = core.prepare_training(gt, label_col, img, log_msg)
                
                # 4. Parameters
                p_sup = dict(train_pct=train_pct, by_group=True, max_per_class=5000, seed=42, 
                             n_trees=200, epochs=40, min_conf=0.40, max_svm=8000)
                p_uns = dict(kmeans=dict(n_classes=8, iterations=20), 
                             isodata=unsup.ISODATA_DEFAULTS, 
                             extract_sig=True, seed=42)
                
                methods = [(k, models.METHODS[k]) for k in selected_methods]
                uns_methods = []
                if run_kmeans: uns_methods.append(("KMEANS", "K-Means"))
                if run_isodata: uns_methods.append(("ISODATA", "ISODATA"))

                # 5. Run Models
                results_sup = None
                results_uns = None
                
                # Use a temp dir for outputs
                out_dir = tempfile.mkdtemp()
                
                if methods:
                    results_sup = core.run_classification(img, prep, methods, p_sup, out_dir, "crop", log_msg)
                if uns_methods:
                    results_uns = core.run_unsupervised(img, uns_methods, p_uns, out_dir, "crop", log_msg)

                # --- DISPLAY RESULTS ---
                st.success("Classification Complete!")
                
                if results_sup:
                    st.header("Supervised Results")
                    tabs = st.tabs([res['label'] for res in results_sup['results'].values()])
                    
                    for idx, (key, res) in enumerate(results_sup['results'].items()):
                        with tabs[idx]:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Plot Map
                                fig, ax = plt.subplots(figsize=(10, 7))
                                arr = res['map']
                                cmap = ListedColormap([tuple(v/255 for v in c) for c in results_sup['colors']])
                                extent = core.extent_of(img) if hasattr(core, 'extent_of') else (img['transform'].c, img['transform'].c + img['W']*img['transform'].a, img['transform'].f + img['H']*img['transform'].e, img['transform'].f)
                                
                                ax.imshow(np.ma.masked_equal(arr, 0), cmap=cmap, vmin=0.5, vmax=len(results_sup['names'])+0.5, extent=extent)
                                ax.set_title(f"{res['label']} - OA: {res['acc']['oa']*100:.1f}%")
                                st.pyplot(fig)
                                
                            with col2:
                                st.metric("Overall Accuracy", f"{res['acc']['oa']*100:.1f}%")
                                st.metric("Kappa", f"{res['acc']['kappa']:.3f}")
                                
                                # Area Table
                                area_data = []
                                for name, area_ha in zip(results_sup['names'], res['class_areas_ha']):
                                    area_data.append({"Class": name, "Area (ha)": f"{area_ha:,.2f}"})
                                st.table(pd.DataFrame(area_data))
                                
                if results_uns:
                    st.header("Unsupervised Results")
                    tabs = st.tabs([res['label'] for res in results_uns['results'].values()])
                    
                    for idx, (key, res) in enumerate(results_uns['results'].items()):
                        with tabs[idx]:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                fig, ax = plt.subplots(figsize=(10, 7))
                                arr = res['map']
                                cmap = ListedColormap([tuple(v/255 for v in c) for c in res['colors']])
                                extent = core.extent_of(img) if hasattr(core, 'extent_of') else (img['transform'].c, img['transform'].c + img['W']*img['transform'].a, img['transform'].f + img['H']*img['transform'].e, img['transform'].f)
                                
                                ax.imshow(np.ma.masked_equal(arr, 0), cmap=cmap, vmin=0.5, vmax=len(res['names'])+0.5, extent=extent)
                                ax.set_title(f"{res['label']}")
                                st.pyplot(fig)
                            with col2:
                                area_data = []
                                for name, area_ha in zip(res['names'], res['class_areas_ha']):
                                    area_data.append({"Class": name, "Area (ha)": f"{area_ha:,.2f}"})
                                st.table(pd.DataFrame(area_data))
                                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)
