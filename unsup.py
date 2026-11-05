"""Unsupervised classification: K-Means and ISODATA.

Both work on the valid pixels of the clipped stack. Clusters are learnt on a random
sample of pixels (bands standardised) and then every pixel is assigned to the nearest
cluster centre, so large districts stay fast.
"""
import numpy as np
import pandas as pd

CHUNK = 200_000

# defaults used by the GUI ("parameters as the model requires")
KMEANS_DEFAULTS = dict(n_classes=8, iterations=20)
ISODATA_DEFAULTS = dict(
    init_clusters=10,      # starting number of clusters
    min_clusters=5,        # never merge below this
    max_clusters=15,       # never split above this
    iterations=10,         # maximum iterations
    min_samples_pct=1.0,   # a cluster smaller than this % of the sample is dissolved
    max_std=1.0,           # split a cluster if its std-dev in any band exceeds this (standardised units)
    min_dist=0.5,          # merge two clusters whose centres are closer than this (standardised units)
    max_merge_pairs=2,     # max number of cluster pairs merged per iteration
    change_pct=5.0,        # stop when fewer than this % of samples change cluster
)


def _assign(X, C):
    """Index of the nearest centre for every row of X (chunked, squared euclidean)."""
    c2 = (C ** 2).sum(1)
    out = np.empty(len(X), np.int32)
    for s in range(0, len(X), 50_000):
        x = X[s:s + 50_000]
        d = (x ** 2).sum(1)[:, None] - 2 * x @ C.T + c2[None, :]
        out[s:s + 50_000] = d.argmin(1)
    return out


def _sample(img, n, seed):
    valid = img["valid"]
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        raise ValueError("No valid pixels inside the selected area.")
    rng = np.random.default_rng(seed)
    if len(rows) > n:
        sel = rng.choice(len(rows), n, replace=False)
        rows, cols = rows[sel], cols[sel]
    X = img["stack"][:, rows, cols].T.astype(np.float32)
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0) + 1e-6
    return np.nan_to_num((X - mu) / sd).astype(np.float32), mu, sd, rows, cols


# ------------------------------------------------------------------ K-Means
def kmeans_centers(Xn, k, iters, seed=42):
    from sklearn.cluster import KMeans
    if len(Xn) < k:
        raise ValueError("Fewer valid pixels than requested classes.")
    km = KMeans(n_clusters=k, init="k-means++", n_init=3, max_iter=iters, tol=1e-4,
                algorithm="lloyd", random_state=seed).fit(Xn)
    return km.cluster_centers_.astype(np.float32), int(km.n_iter_), km.n_iter_ < iters


# ------------------------------------------------------------------ ISODATA
def isodata_centers(Xn, p, seed=42, log=print):
    rng = np.random.default_rng(seed)
    K0 = int(p["init_clusters"])
    kmin, kmax = int(p["min_clusters"]), int(p["max_clusters"])
    if not (kmin <= K0 <= kmax):
        raise ValueError("ISODATA: need  min clusters <= initial clusters <= max clusters.")
    min_n = max(2, int(len(Xn) * p["min_samples_pct"] / 100.0))
    # initial centres: k-means++ style seeding
    C = [Xn[rng.integers(len(Xn))]]
    for _ in range(K0 - 1):
        d = ((Xn[:, None, :] - np.array(C)[None]) ** 2).sum(2).min(1) if len(Xn) <= 20000 else \
            np.min([((Xn - c) ** 2).sum(1) for c in C], axis=0)
        C.append(Xn[rng.choice(len(Xn), p=d / d.sum())] if d.sum() > 0 else Xn[rng.integers(len(Xn))])
    C = np.array(C, np.float32)
    prev, converged, it_done = None, False, 0

    def recompute(C, lab):
        """drop small clusters, update means, return (C, lab, stds, counts)."""
        keep = [i for i in range(len(C)) if (lab == i).sum() >= min_n]
        if len(keep) < 2:                      # never collapse completely
            keep = list(np.argsort([-(lab == i).sum() for i in range(len(C))])[:2])
        C = C[keep]
        lab = _assign(Xn, C)
        cnt = np.bincount(lab, minlength=len(C))
        newC = np.array([Xn[lab == i].mean(0) if cnt[i] else C[i] for i in range(len(C))], np.float32)
        std = np.array([Xn[lab == i].std(0) if cnt[i] > 1 else np.zeros(Xn.shape[1]) for i in range(len(C))])
        return newC, lab, std, cnt

    for it in range(1, int(p["iterations"]) + 1):
        it_done = it
        lab = _assign(Xn, C)
        n_before = len(C)
        C, lab, std, cnt = recompute(C, lab)
        dropped = len(C) != n_before
        changed = 100.0 if (prev is None or dropped) else 100.0 * (prev != lab).mean()
        prev = lab.copy()
        nc = len(C)
        acted = dropped
        last = it == int(p["iterations"])
        if not last:
            # ---- split
            if nc <= K0 / 2 or (it % 2 == 1 and nc < 2 * K0):
                newC = []
                for i in range(nc):
                    j = int(std[i].argmax())
                    if std[i, j] > p["max_std"] and cnt[i] >= 2 * min_n and nc + len(newC) < kmax:
                        a, b = C[i].copy(), C[i].copy()
                        a[j] += 0.5 * std[i, j]
                        b[j] -= 0.5 * std[i, j]
                        C[i] = a
                        newC.append(b)
                if newC:
                    C = np.vstack([C] + newC)
                    acted = True
            # ---- merge
            elif nc > kmin and (it % 2 == 0 or nc >= 2 * K0):
                D = np.sqrt(((C[:, None] - C[None]) ** 2).sum(2))
                iu = np.triu_indices(nc, 1)
                order = np.argsort(D[iu])
                used, merged, new = set(), 0, []
                for o in order:
                    i, j = iu[0][o], iu[1][o]
                    if D[i, j] >= p["min_dist"] or merged >= p["max_merge_pairs"] or nc - merged <= kmin:
                        break
                    if i in used or j in used:
                        continue
                    w = cnt[i] + cnt[j]
                    new.append((cnt[i] * C[i] + cnt[j] * C[j]) / max(w, 1))
                    used.update((i, j))
                    merged += 1
                if merged:
                    C = np.vstack([C[[i for i in range(nc) if i not in used]]] + [np.array(new)])
                    acted = True
        if acted:
            prev = None                      # cluster indices changed -> restart change counting
        log(f"    ISODATA iter {it}: {len(C)} clusters, {changed:.1f}% pixels changed"
            + ("  (split/merge)" if acted else ""))
        if it > 1 and changed < p["change_pct"] and not acted:
            converged = True
            break
    # enforce [min, max] clusters on the final set
    lab = _assign(Xn, C)
    C, lab, std, cnt = recompute(C, lab)
    while len(C) > kmax:
        D = np.sqrt(((C[:, None] - C[None]) ** 2).sum(2)) + np.eye(len(C)) * 1e9
        i, j = np.unravel_index(D.argmin(), D.shape)
        C[i] = (cnt[i] * C[i] + cnt[j] * C[j]) / max(cnt[i] + cnt[j], 1)
        C = np.delete(C, j, 0)
        lab = _assign(Xn, C)
        cnt = np.bincount(lab, minlength=len(C))
    return C.astype(np.float32), it_done, converged


# ------------------------------------------------------------------ full map + signatures
def run(img, method, params, seed=42, log=print, sample_size=200_000):
    """Returns dict(map uint8 (0 = nodata, 1..n classes), n_classes, n_iter, converged)."""
    Xn, mu, sd, _, _ = _sample(img, sample_size, seed)
    if method == "KMEANS":
        C, n_iter, conv = kmeans_centers(Xn, int(params["n_classes"]), int(params["iterations"]), seed)
    else:
        C, n_iter, conv = isodata_centers(Xn, params, seed, log)
    # order clusters by mean value over all bands (original units) so class 1 = lowest
    order = np.argsort((C * sd + mu).mean(1))
    C = C[order]
    B, H, W = img["stack"].shape
    flat = img["stack"].reshape(B, -1)
    idx = np.flatnonzero(img["valid"].ravel())
    out = np.zeros(H * W, np.uint8)
    for s in range(0, len(idx), CHUNK):
        ii = idx[s:s + CHUNK]
        X = np.nan_to_num((flat[:, ii].T - mu) / sd).astype(np.float32)
        out[ii] = _assign(X, C) + 1
    return dict(map=out.reshape(H, W), n_classes=len(C), n_iter=n_iter, converged=bool(conv))


def signatures(img, final, band_names, max_px=2_000_000, seed=42):
    """Per-class mean / std / min / max of every band (original units), as DataFrames."""
    rows, cols = np.nonzero(final)
    if len(rows) > max_px:
        sel = np.random.default_rng(seed).choice(len(rows), max_px, replace=False)
        rows, cols = rows[sel], cols[sel]
    df = pd.DataFrame(img["stack"][:, rows, cols].T, columns=band_names)
    df["Class"] = [f"Class {c}" for c in final[rows, cols]]
    g = df.groupby("Class", sort=False)
    order = sorted(df["Class"].unique(), key=lambda s: int(s.split()[1]))
    return {"Mean": g.mean().loc[order], "Std Dev": g.std().fillna(0).loc[order],
            "Min": g.min().loc[order], "Max": g.max().loc[order], "n_used": len(rows)}
 
