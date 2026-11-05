"""Data handling, sampling, accuracy and export (no GUI code here)."""
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, geometry_window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import rowcol
import matplotlib

import config

OTHER = config.OTHER_CLASS_NAME


def _noop(*a, **k):
    pass


# --------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------
def find_col(gdf, keywords):
    cols = [c for c in gdf.columns if c != "geometry"]
    for kw in keywords:
        for c in cols:
            if kw in c.lower():
                return c
    for c in cols:
        if gdf[c].dtype == object:
            return c
    return cols[0]


def load_boundaries():
    d = gpd.read_file(config.DISTRICT_SHP)
    t = gpd.read_file(config.TALUKA_SHP)
    if d.crs is None:
        d = d.set_crs(epsg=config.DISTRICT_FALLBACK_EPSG)
    if t.crs is None:
        t = t.set_crs(epsg=config.TALUKA_FALLBACK_EPSG)
    t = t.to_crs(d.crs)
    
    dcol = config.DISTRICT_NAME_FIELD or find_col(d, ["district", "DTENAME", "dtname", "name"])
    tcol = config.TALUKA_NAME_FIELD or find_col(t, ["tahsil", "THENAME", "tehsil", "tahasil", "name"])
    
    # Clean and filter out nulls to prevent sorting errors
    d = d[d[dcol].notna()].copy()
    t = t[t[tcol].notna()].copy()
    
    d[dcol] = d[dcol].astype(str).str.strip()
    t[tcol] = t[tcol].astype(str).str.strip()
    
    return {"district": d, "taluka": t, "dcol": dcol, "tcol": tcol}


def district_names(b):
    return sorted(b["district"][b["dcol"]].unique())


def _talukas_in_district(b, district):
    dgeom = b["district"].loc[b["district"][b["dcol"]] == district].geometry.unary_union
    t = b["taluka"]
    inside = t.geometry.representative_point().within(dgeom.buffer(1.0))
    return t[inside]


def taluka_names(b, district):
    return sorted(_talukas_in_district(b, district)[b["tcol"]].unique())


def select_region(b, district, taluka=None):
    """Returns a 1-row GeoDataFrame (CRS of district file) of the selected area."""
    if taluka:
        sub = _talukas_in_district(b, district)
        sub = sub[sub[b["tcol"]] == taluka]
    else:
        sub = b["district"][b["district"][b["dcol"]] == district]
    if sub.empty:
        raise ValueError("Selected district/taluka not found in boundary file.")
    return gpd.GeoDataFrame(geometry=[sub.geometry.unary_union], crs=sub.crs)


# --------------------------------------------------------------------------
# Image stack
# --------------------------------------------------------------------------
def _same_grid(a, b):
    return (a.crs == b.crs and a.transform == b.transform
            and a.width == b.width and a.height == b.height)


def load_stack(paths, region_gdf, log=_noop):
    """Read all images, align to the first, clip to region. Returns dict."""
    ds0 = rasterio.open(paths[0])
    geom = region_gdf.to_crs(ds0.crs).geometry.unary_union
    try:
        win = geometry_window(ds0, [geom])
    except Exception:
        raise ValueError("Selected district/taluka does not overlap the first image.")
    win = win.round_offsets().round_lengths()
    transform = ds0.window_transform(win)
    H, W = int(win.height), int(win.width)
    log(f"Reading {len(paths)} image(s); clipped window = {H} rows x {W} cols")

    arrs, names, counts = [], [], []
    for p in paths:
        ds = ds0 if p == paths[0] else rasterio.open(p)
        src = ds
        if not _same_grid(ds, ds0):
            log(f"  Resampling {os.path.basename(p)} to the grid of the first image")
            src = WarpedVRT(ds, crs=ds0.crs, transform=ds0.transform, width=ds0.width,
                            height=ds0.height, resampling=Resampling.bilinear)
        a = src.read(window=win).astype("float32")
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        arrs.append(a)
        counts.append(a.shape[0])
        stem = os.path.splitext(os.path.basename(p))[0]
        names += [f"{stem}_b{i + 1}" for i in range(a.shape[0])]
        if src is not ds:
            src.close()
        if ds is not ds0:
            ds.close()
    crs = ds0.crs
    ds0.close()

    stack = np.concatenate(arrs, axis=0)
    inside = rasterize([(geom, 1)], out_shape=(H, W), transform=transform,
                       fill=0, dtype="uint8").astype(bool)
    # a pixel is unusable if any band is NaN or if ALL bands are 0 (=> "0 class")
    valid = inside & ~np.isnan(stack).any(0) & ~(stack == 0).all(0)
    if len(paths) > 1 and len(set(counts)) == 1:
        seq_shape = (len(paths), counts[0])      # (time steps, features per step)
    else:
        seq_shape = (stack.shape[0], 1)
    log(f"Stack: {stack.shape[0]} bands, {int(valid.sum()):,} valid pixels inside the area")
    return dict(stack=stack, valid=valid, transform=transform, crs=crs, paths=list(paths),
                band_names=names, seq_shape=seq_shape, region_geom=geom, H=H, W=W)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def load_gt(path):
    gt = gpd.read_file(path)
    if gt.crs is None:
        raise ValueError("Training data has no CRS (.prj missing).")
    return gt


def prepare_training(gt, label_col, img, log=_noop):
    gt = gt.to_crs(img["crs"]).copy()
    gt["_fid"] = np.arange(len(gt))                 # original row id (0-based) of the shapefile
    gt = gt[gt.geometry.notna() & ~gt.geometry.is_empty]
    n_all = len(gt)
    gt = gpd.clip(gt, img["region_geom"])
    gt = gt[~gt.geometry.is_empty]
    gt = gt.explode(index_parts=False)
    gt = gt[gt[label_col].notna()].copy()
    gt["_cls"] = gt[label_col].astype(str).str.strip()
    log(f"GT features: {n_all} in file, {gt['_fid'].nunique()} fall inside the selected district/taluka")

    valid, tr = img["valid"], img["transform"]
    H, W = valid.shape
    recs, removed = [], []

    pts = gt[gt.geom_type == "Point"]
    if len(pts):
        r, c = rowcol(tr, pts.geometry.x.values, pts.geometry.y.values)
        r, c = np.asarray(r), np.asarray(c)
        inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        ok = inb.copy()
        ok[inb] = valid[r[inb], c[inb]]
        for fid, cl, rr, cc in zip(pts["_fid"].values[ok], pts["_cls"].values[ok], r[ok], c[ok]):
            recs.append((rr, cc, fid, cl, "point"))
        for fid, cl, i in zip(pts["_fid"].values[~ok], pts["_cls"].values[~ok], inb[~ok]):
            removed.append((fid, cl, "point", "outside image extent" if not i else "on 0/NoData pixel"))

    pol = gt[gt.geom_type.isin(["Polygon", "MultiPolygon"])]
    if len(pol):
        local = np.arange(1, len(pol) + 1, dtype="int32")
        lab = rasterize(list(zip(pol.geometry, local)), out_shape=(H, W), transform=tr,
                        fill=0, dtype="int32")
        rr, cc = np.nonzero(lab)
        k = lab[rr, cc] - 1
        m = valid[rr, cc]
        rr, cc, k = rr[m], cc[m], k[m]
        fids, clss = pol["_fid"].values, pol["_cls"].values
        recs += list(zip(rr, cc, fids[k], clss[k], ["polygon"] * len(rr)))
        have = set(np.unique(k).tolist())
        for i in range(len(pol)):
            if i not in have:
                removed.append((fids[i], clss[i], "polygon", "no valid pixels (outside image / 0 / NoData)"))

    if not recs:
        raise ValueError("No training samples fall on valid image pixels.")
    samples = pd.DataFrame(recs, columns=["row", "col", "fid", "cls", "kind"])
    classes = sorted(samples["cls"].unique())
    samples["y"] = samples["cls"].map({c: i for i, c in enumerate(classes)})
    rem = pd.DataFrame(removed, columns=["ID", "Class", "Type", "Reason"]).drop_duplicates()
    # a polygon id may appear once per polygon-part; keep unique (ID, Reason)
    if len(rem):
        ids = rem["ID"].tolist()
        log(f"REMOVED {len(rem)} feature(s) not on valid image pixels. IDs: {ids}")
        for _, x in rem.iterrows():
            log(f"   id={x.ID}  class={x.Class}  ({x.Type}): {x.Reason}")
    else:
        log("All GT features fall on valid image pixels.")

    counts = (samples.groupby("cls").agg(Features=("fid", "nunique"), Pixels=("fid", "size"))
              .reindex(classes).reset_index().rename(columns={"cls": "Class"}))
    return dict(gt=gt, samples=samples, classes=classes, removed=rem, counts=counts)


def split_samples(samples, train_frac, by_group=True, max_train=0, seed=42):
    rng = np.random.default_rng(seed)
    tr_all, te_all = [], []
    for _, g in samples.groupby("y"):
        idx = g.index.values
        fids = g["fid"].unique()
        if by_group and len(fids) >= 2:
            fids = rng.permutation(fids)
            n = int(np.clip(round(len(fids) * train_frac), 1, len(fids) - 1))
            m = g["fid"].isin(set(fids[:n])).values
            tr, te = idx[m], idx[~m]
        else:
            idx = rng.permutation(idx)
            n = max(1, int(round(len(idx) * train_frac)))
            if len(idx) > 1:
                n = min(n, len(idx) - 1)
            tr, te = idx[:n], idx[n:]
        if max_train and len(tr) > max_train:
            tr = rng.choice(tr, max_train, replace=False)
        tr_all.append(tr)
        te_all.append(te)
    return np.concatenate(tr_all), np.concatenate(te_all)


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------
def accuracy(y_true, y_pred, names):
    n = len(names)
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tot = cm.sum()
    diag = np.diag(cm).astype(float)
    row, col = cm.sum(1).astype(float), cm.sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pa, ua = diag / row, diag / col
        oa = diag.sum() / tot if tot else np.nan
        pe = (row * col).sum() / (tot ** 2) if tot else np.nan
        kappa = (oa - pe) / (1 - pe) if tot and pe < 1 else np.nan
    return dict(cm=cm, oa=oa, kappa=kappa, pa=pa, ua=ua, names=list(names), n=int(tot))


# --------------------------------------------------------------------------
# Colours / outputs
# --------------------------------------------------------------------------
def class_colors(k):
    cmap = matplotlib.colormaps["tab20"]
    cols = [tuple(int(255 * v) for v in cmap(i % 20)[:3]) for i in range(k)]
    return cols + [(128, 128, 128)]                  # grey = Other crop


def write_geotiff(path, arr, img, names, colors):
    prof = dict(driver="GTiff", dtype="uint8", count=1, width=arr.shape[1], height=arr.shape[0],
                crs=img["crs"], transform=img["transform"], nodata=0, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)
        dst.write_colormap(1, {i + 1: colors[i] + (255,) for i in range(len(names))})
        dst.update_tags(**{f"class_{i + 1}": n for i, n in enumerate(names)})
    pd.DataFrame({"Code": range(1, len(names) + 1), "Class": names}).to_csv(
        os.path.splitext(path)[0] + "_legend.csv", index=False)


def write_excel(path, results, prep, names, tr_idx, te_idx):
    s = prep["samples"]
    counts = prep["counts"].copy()
    counts["Train_pixels"] = counts["Class"].map(s.loc[tr_idx, "cls"].value_counts()).fillna(0).astype(int)
    counts["Test_pixels"] = counts["Class"].map(s.loc[te_idx, "cls"].value_counts()).fillna(0).astype(int)
    
    # Summary of accuracy across models
    summ = pd.DataFrame([{
        "Method": r["label"], 
        "Train OA (%)": r["acc_train"]["oa"] * 100,
        "Train Kappa": r["acc_train"]["kappa"], 
        "Test OA (%)": r["acc"]["oa"] * 100,
        "Test Kappa": r["acc"]["kappa"], 
        "Train N": r["acc_train"]["n"], 
        "Test N": r["acc"]["n"]
    } for r in results.values()])

    # Build Class-wise Area Summary Table across all methods
    area_summary_data = {"Class": names}
    for key, r in results.items():
        px = np.asarray(r["class_pixel_counts"], dtype=float)
        ha = np.asarray(r["class_areas_ha"], dtype=float)
        area_summary_data[f"{r['label']} - Pixels"] = list(px.astype(np.int64)) + [int(px.sum())]
        area_summary_data[f"{r['label']} - Area (ha)"] = list(np.round(ha, 2)) + [round(ha.sum(), 2)]
        area_summary_data[f"{r['label']} - Area (%)"] = list(np.round(100 * ha / ha.sum(), 2)) + [100.0]
    area_summary_data["Class"] = list(names) + ["TOTAL"]
    area_df = pd.DataFrame(area_summary_data)

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summ.round(4).to_excel(xw, sheet_name="Summary", index=False)
        area_df.to_excel(xw, sheet_name="Area_Statistics_ha", index=False)
        counts.to_excel(xw, sheet_name="Training_counts", index=False)
        prep["removed"].to_excel(xw, sheet_name="Removed_GT", index=False)
        
        for key, r in results.items():
            a = r["acc"]
            sh = key[:31]
            
            # Write metrics summary
            pd.DataFrame({
                "Metric": ["Overall Accuracy (%)", "Kappa coefficient", "Test samples"],
                "Value": [a["oa"] * 100, a["kappa"], a["n"]]
            }).round(4).to_excel(xw, sheet_name=sh, index=False, startrow=0)
            
            # Write Error / Confusion Matrix
            cm = pd.DataFrame(a["cm"], index=[f"Ref: {n}" for n in names],
                              columns=[f"Cls: {n}" for n in names])
            cm["Total (ref)"] = cm.sum(1)
            cm["Producer Acc (%)"] = a["pa"] * 100
            cm.loc["Total (classified)"] = list(a["cm"].sum(0)) + [a["cm"].sum(), np.nan]
            cm.loc["User Acc (%)"] = list(a["ua"] * 100) + [np.nan, a["oa"] * 100]
            
            pd.Series(["ERROR MATRIX (rows = reference, columns = classified)"]).to_excel(
                xw, sheet_name=sh, startrow=5, index=False, header=False)
            cm.round(2).to_excel(xw, sheet_name=sh, startrow=6)
            
            # Write Producer & User Accuracy
            cr = len(names) + 11
            pd.DataFrame({
                "Class": names, 
                "Producer Accuracy (%)": a["pa"] * 100,
                "User Accuracy (%)": a["ua"] * 100
            }).round(2).to_excel(xw, sheet_name=sh, startrow=cr, index=False)

            # Write Method Specific Class-wise Area Table
            ar = cr + len(names) + 3
            pd.Series([f"CLASSIFIED AREA STATISTICS ({r['label']})"]).to_excel(
                xw, sheet_name=sh, startrow=ar, index=False, header=False)
            px = np.asarray(r["class_pixel_counts"], dtype=float)
            ha = np.asarray(r["class_areas_ha"], dtype=float)
            pd.DataFrame({
                "Class": list(names) + ["TOTAL"],
                "Pixel Count": list(px.astype(np.int64)) + [int(px.sum())],
                "Area (ha)": list(np.round(ha, 2)) + [round(ha.sum(), 2)],
                "Area (%)": list(np.round(100 * ha / ha.sum(), 2)) + [100.0],
            }).to_excel(xw, sheet_name=sh, startrow=ar + 1, index=False)


def safe_write_excel(path, results, prep, names, tr_idx, te_idx, log=_noop):
    """Write the workbook; if the file is locked (e.g. open in Excel) or not writable,
    fall back to a time-stamped file name instead of crashing the whole run."""
    base, ext = os.path.splitext(path)
    alt = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
    for target in (path, alt):
        try:
            write_excel(target, results, prep, names, tr_idx, te_idx)
            if target != path:
                log(f"WARNING: '{path}' is locked (is it open in Excel?). Saved instead as: {target}")
            else:
                log(f"Accuracy workbook saved: {target}")
            return target
        except PermissionError:
            continue
    log("ERROR: could not write the Excel file - close it in Excel or choose a folder you can write to. "
        "The classified .tif files were saved correctly.")
    return None


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run_classification(img, prep, methods, params, out_dir, tag, log=_noop):
    import models
    os.makedirs(out_dir, exist_ok=True)
    classes = prep["classes"]
    K = len(classes)
    names = classes + [OTHER]
    colors = class_colors(K)
    S = prep["samples"]
    tr_idx, te_idx = split_samples(S, params["train_pct"] / 100.0, params["by_group"],
                                   params["max_per_class"], params["seed"])
    log(f"Samples: train = {len(tr_idx):,} px, test = {len(te_idx):,} px "
        f"({len(classes)} classes + '{OTHER}')")
    tr, te = S.loc[tr_idx], S.loc[te_idx]
    
    # Calculate pixel resolution & single-pixel area in hectares (ha)
    tr_transform = img["transform"]
    pixel_area_m2 = abs(tr_transform.a * tr_transform.e)
    pixel_area_ha = pixel_area_m2 / 10000.0

    results = {}
    for key, label in methods:
        t0 = time.time()
        log(f"--- {label}: training ...")
        model = models.build_model(key, K, img["seq_shape"], params, log)
        model.fit(img["stack"], tr.row.values, tr.col.values, tr.y.values)
        log(f"--- {label}: classifying full area ...")
        pred, conf = model.predict_map(img["stack"], img["valid"])
        final = (pred + 1).astype(np.uint8)
        final[conf < params["min_conf"]] = K + 1          # not recognised -> Other crop
        final[~img["valid"]] = 0

        # Calculate class-wise pixel counts & area in ha
        # final maps use values 1 to len(names)
        class_px_counts = [(final == (i + 1)).sum() for i in range(len(names))]
        class_areas_ha = [count * pixel_area_ha for count in class_px_counts]

        def ev(df):
            p = final[df.row.values, df.col.values].astype(int) - 1
            return accuracy(df.y.values, p, names), p

        acc_te, p_te = ev(te)
        acc_tr, _ = ev(tr)
        tif = os.path.join(out_dir, f"{tag}_{key}.tif")
        write_geotiff(tif, final, img, names, colors)
        log(f"    {label}: Test OA = {acc_te['oa'] * 100:.2f}%  Kappa = {acc_te['kappa']:.3f}  "
            f"(train OA {acc_tr['oa'] * 100:.2f}%)  [{time.time() - t0:.0f}s]  -> {tif}")
        
        results[key] = dict(
            label=label, map=final, acc=acc_te, acc_train=acc_tr, tif=tif,
            te_rows=te.row.values, te_cols=te.col.values,
            te_correct=(p_te == te.y.values),
            class_pixel_counts=class_px_counts,
            class_areas_ha=class_areas_ha
        )

    xl = os.path.join(out_dir, f"{tag}_accuracy_assessment.xlsx")
    xl = safe_write_excel(xl, results, prep, names, tr_idx, te_idx, log)
    return dict(results=results, names=names, colors=colors, excel=xl)

# --------------------------------------------------------------------------
# Unsupervised classification (K-Means / ISODATA)
# --------------------------------------------------------------------------
def unsup_colors(n):
    cmap = matplotlib.colormaps["tab20" if n <= 20 else "turbo"]
    return [tuple(int(255 * v) for v in cmap((i % 20) / 20 if n <= 20 else i / max(n - 1, 1))[:3])
            for i in range(n)]


def write_excel_unsup(path, results, extract_sig):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame([{
            "Method": r["label"], "Classes": r["n_classes"], "Iterations used": r["n_iter"],
            "Converged": "Yes" if r["converged"] else "No (max iterations reached)",
            "Mapped pixels": int(sum(r["class_pixel_counts"])),
            "Mapped area (ha)": round(sum(r["class_areas_ha"]), 2),
            "Parameters": ", ".join(f"{k}={v}" for k, v in r["params"].items())}
            for r in results.values()]).to_excel(xw, sheet_name="Summary", index=False)
        # area comparison (ha) of every class for each method
        for key, r in results.items():
            px = np.asarray(r["class_pixel_counts"], float)
            ha = np.asarray(r["class_areas_ha"], float)
            pd.DataFrame({
                "Class": list(r["names"]) + ["TOTAL"],
                "Pixel Count": list(px.astype(np.int64)) + [int(px.sum())],
                "Area (ha)": list(np.round(ha, 2)) + [round(ha.sum(), 2)],
                "Area (%)": list(np.round(100 * ha / max(ha.sum(), 1e-9), 2)) + [100.0],
            }).to_excel(xw, sheet_name=f"Area_{key}"[:31], index=False)
        if extract_sig:
            for key, r in results.items():
                sig = r["signatures"]
                sh = f"Signatures_{key}"[:31]
                row = 0
                pd.Series([f"CLASS SIGNATURES - {r['label']}  (pixel values of each input band; "
                           f"statistics from {sig['n_used']:,} pixels)"]).to_excel(
                    xw, sheet_name=sh, startrow=row, index=False, header=False)
                row += 2
                for stat in ("Mean", "Std Dev", "Min", "Max"):
                    pd.Series([stat]).to_excel(xw, sheet_name=sh, startrow=row, index=False, header=False)
                    sig[stat].round(5).to_excel(xw, sheet_name=sh, startrow=row + 1)
                    row += len(sig[stat]) + 4


def safe_write_excel_unsup(path, results, extract_sig, log=_noop):
    base, ext = os.path.splitext(path)
    alt = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
    for target in (path, alt):
        try:
            write_excel_unsup(target, results, extract_sig)
            log(("WARNING: file locked (open in Excel?). Saved instead as: " if target != path
                 else "Excel saved: ") + target)
            return target
        except PermissionError:
            continue
    log("ERROR: could not write the Excel file - close it in Excel or choose another folder.")
    return None


def run_unsupervised(img, methods, params, out_dir, tag, log=_noop):
    """methods: list of (key, label) with key in {'KMEANS', 'ISODATA'}.
    params: dict(kmeans={...}, isodata={...}, extract_sig=bool, seed=int)"""
    import unsup
    os.makedirs(out_dir, exist_ok=True)
    pixel_area_ha = abs(img["transform"].a * img["transform"].e) / 10000.0
    if img["crs"].is_geographic:
        log("WARNING: image CRS is geographic - hectare areas are not meaningful; use a projected CRS.")
    results = {}
    for key, label in methods:
        t0 = time.time()
        p = params["kmeans"] if key == "KMEANS" else params["isodata"]
        log(f"--- {label}: clustering ...")
        r = unsup.run(img, key, p, params.get("seed", 42), log)
        final, n = r["map"], r["n_classes"]
        names = [f"Class {i + 1}" for i in range(n)]
        colors = unsup_colors(n)
        counts = np.bincount(final.ravel(), minlength=n + 1)[1:n + 1]
        sig = unsup.signatures(img, final, img["band_names"])
        tif = os.path.join(out_dir, f"{tag}_{key}.tif")
        write_geotiff(tif, final, img, names, colors)
        log(f"    {label}: {n} classes, {r['n_iter']} iteration(s), "
            f"{'converged' if r['converged'] else 'max iterations reached'}  [{time.time() - t0:.0f}s]  -> {tif}")
        results[key] = dict(label=label, map=final, names=names, colors=colors, n_classes=n,
                            n_iter=r["n_iter"], converged=r["converged"], params=dict(p), tif=tif,
                            class_pixel_counts=[int(c) for c in counts],
                            class_areas_ha=[float(c * pixel_area_ha) for c in counts],
                            signatures=sig, band_names=list(img["band_names"]))
    xl = os.path.join(out_dir, f"{tag}_unsupervised.xlsx")
    xl = safe_write_excel_unsup(xl, results, params.get("extract_sig", True), log)
    return dict(results=results, excel=xl)
