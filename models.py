"""Classifiers. Every model offers:
    fit(stack, rows, cols, y)             y = class index 0..K-1
    predict_map(stack, valid) -> (pred (H,W) class index, conf (H,W) 0..1)
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:                                   # classical models still work
    HAS_TORCH = False

METHODS = {                                           # key -> GUI label
    "RF": "Random Forest (RF)",
    "XGB": "XGBoost",
    "SVM": "SVM (RBF)",
    "DT": "Decision Tree (DT)",
    "MLC": "Maximum Likelihood (MLC)",
    "CNN": "CNN (1-D, pixel)",
    "LSTM": "LSTM",
    "UNET": "U-Net",
    "TRANS": "Transformer",
}
CHUNK = 200_000


# ============================================================ pixel models
class PixelModel:
    def __init__(self, K, seq_shape, params, log):
        self.K, self.seq_shape, self.p, self.log = K, seq_shape, params, log

    def fit(self, stack, rows, cols, y):
        X = stack[:, rows, cols].T.astype(np.float32)
        self.mu = np.nanmean(X, 0)
        self.sd = np.nanstd(X, 0) + 1e-6
        self.classes_ = np.unique(y)
        self._fit(self._norm(X), np.searchsorted(self.classes_, y))

    def _norm(self, X):
        return np.nan_to_num((X - self.mu) / self.sd).astype(np.float32)

    def predict_map(self, stack, valid):
        B, H, W = stack.shape
        flat = stack.reshape(B, -1)
        idx = np.flatnonzero(valid.ravel())
        pred = np.zeros(H * W, np.int16)
        conf = np.zeros(H * W, np.float32)
        for s in range(0, len(idx), CHUNK):
            ii = idx[s:s + CHUNK]
            k, c = self._predict(self._norm(flat[:, ii].T))
            pred[ii] = self.classes_[k]
            conf[ii] = c
        return pred.reshape(H, W), conf.reshape(H, W)


class SkModel(PixelModel):
    def _make(self):
        raise NotImplementedError

    def _fit(self, X, y):
        self.clf = self._make()
        self.clf.fit(X, y)

    def _predict(self, X):
        pr = self.clf.predict_proba(X)
        return pr.argmax(1), pr.max(1)


class RF(SkModel):
    def _make(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=self.p["n_trees"], n_jobs=-1,
                                      class_weight="balanced", random_state=42)


class DT(SkModel):
    def _make(self):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(class_weight="balanced", random_state=42)


class XGB(SkModel):
    def _make(self):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost is not installed: pip install xgboost")
        return XGBClassifier(n_estimators=self.p["n_trees"], max_depth=6, learning_rate=0.1,
                             n_jobs=-1, tree_method="hist")


class SVM(PixelModel):
    def _fit(self, X, y):
        from sklearn.svm import SVC
        cap = self.p["max_svm"]
        if len(X) > cap:
            sel = np.random.default_rng(42).choice(len(X), cap, replace=False)
            X, y = X[sel], y[sel]
            self.log(f"    SVM: sub-sampled training set to {cap} pixels (speed)")
        self.clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
        self.clf.fit(X, y)
        self.log("    SVM: prediction over the full map can be slow for large areas")

    def _predict(self, X):
        k = self.clf.predict(X).astype(int)
        return k, np.ones(len(k), np.float32)         # no probability -> never 'Other'


class MLC(PixelModel):
    """Gaussian Maximum Likelihood Classifier (equal priors)."""
    def _fit(self, X, y):
        d = X.shape[1]
        self.m, self.ic, self.ld = [], [], []
        for k in range(len(self.classes_)):
            Xk = X[y == k]
            C = np.cov(Xk, rowvar=False) if len(Xk) > 1 else np.zeros((d, d))
            C = np.atleast_2d(C) + 1e-2 * np.eye(d)
            self.m.append(Xk.mean(0))
            self.ic.append(np.linalg.inv(C))
            self.ld.append(np.linalg.slogdet(C)[1])

    def _predict(self, X):
        ll = np.stack([-0.5 * (self.ld[k] + np.einsum("ij,jk,ik->i", X - self.m[k], self.ic[k], X - self.m[k]))
                       for k in range(len(self.m))], 1)
        ll -= ll.max(1, keepdims=True)
        pr = np.exp(ll)
        pr /= pr.sum(1, keepdims=True)
        return pr.argmax(1), pr.max(1)


# ============================================================ torch nets
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if HAS_TORCH:
    class CNNNet(nn.Module):                          # 1-D conv along the band / time axis
        def __init__(self, B, K, seq):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
            self.h = nn.Sequential(nn.Flatten(), nn.Linear(64 * B, 128), nn.ReLU(),
                                   nn.Dropout(0.3), nn.Linear(128, K))

        def forward(self, x):
            return self.h(self.f(x.unsqueeze(1)))

    class LSTMNet(nn.Module):
        def __init__(self, B, K, seq):
            super().__init__()
            self.T, self.F = seq
            self.rnn = nn.LSTM(self.F, 64, num_layers=2, batch_first=True, dropout=0.2)
            self.o = nn.Linear(64, K)

        def forward(self, x):
            h, _ = self.rnn(x.view(len(x), self.T, self.F))
            return self.o(h[:, -1])

    class TransNet(nn.Module):
        def __init__(self, B, K, seq):
            super().__init__()
            self.T, self.F = seq
            d = 32
            self.emb = nn.Linear(self.F, d)
            self.pos = nn.Parameter(torch.zeros(1, self.T, d))
            layer = nn.TransformerEncoderLayer(d, 4, 64, 0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
            self.o = nn.Linear(d, K)

        def forward(self, x):
            z = self.emb(x.view(len(x), self.T, self.F)) + self.pos
            return self.o(self.enc(z).mean(1))

    def _block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

    class UNetNet(nn.Module):
        def __init__(self, cin, K, b=16):
            super().__init__()
            self.e1, self.e2, self.e3 = _block(cin, b), _block(b, 2 * b), _block(2 * b, 4 * b)
            self.pool = nn.MaxPool2d(2)
            self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), _block(4 * b, 2 * b)
            self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), _block(2 * b, b)
            self.out = nn.Conv2d(b, K, 1)

        def forward(self, x):                         # H, W must be multiples of 4
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)


def _class_weights(y, K):
    c = np.bincount(y, minlength=K).astype(np.float32)
    w = c.sum() / (K * np.maximum(c, 1))
    return torch.tensor(w, device=_device())


class TorchPixelModel(PixelModel):
    NET = None

    def _fit(self, X, y):
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed: pip install torch")
        dev, K, ep = _device(), len(self.classes_), self.p["epochs"]
        self.net = self.NET(X.shape[1], K, self.seq_shape).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        lossf = nn.CrossEntropyLoss(weight=_class_weights(y, K))
        Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.long)
        self.log(f"    training on {dev}, {ep} epochs, {len(X)} pixels")
        for e in range(ep):
            self.net.train()
            perm, tot = torch.randperm(len(Xt)), 0.0
            for s in range(0, len(Xt), 256):
                b = perm[s:s + 256]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = lossf(self.net(Xt[b].to(dev)), yt[b].to(dev))
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            if (e + 1) % 10 == 0 or e == 0:
                self.log(f"    epoch {e + 1}/{ep}  loss {tot / len(Xt):.4f}")

    def _predict(self, X):
        self.net.eval()
        ks, cs = [], []
        with torch.no_grad():
            for s in range(0, len(X), 8192):
                pr = torch.softmax(self.net(torch.tensor(X[s:s + 8192]).to(_device())), 1)
                c, k = pr.max(1)
                ks.append(k.cpu().numpy())
                cs.append(c.cpu().numpy())
        return np.concatenate(ks), np.concatenate(cs)


class CNN(TorchPixelModel):
    NET = CNNNet if HAS_TORCH else None


class LSTM(TorchPixelModel):
    NET = LSTMNet if HAS_TORCH else None


class TRANS(TorchPixelModel):
    NET = TransNet if HAS_TORCH else None


# ============================================================ U-Net
class UNET:
    """Patch-based U-Net trained on sparse GT pixels (unlabelled pixels are ignored in the loss)."""
    P = 32

    def __init__(self, K, seq_shape, params, log):
        self.K, self.p, self.log = K, params, log

    def _n(self, a):
        return np.nan_to_num((a - self.mu[:, None, None]) / self.sd[:, None, None]).astype(np.float32)

    def fit(self, stack, rows, cols, y):
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed: pip install torch")
        B, H, W = stack.shape
        P, dev = self.P, _device()
        self.classes_ = np.unique(y)
        ye = np.searchsorted(self.classes_, y)
        step = max(1, (H * W) // 500_000)
        smp = stack.reshape(B, -1)[:, ::step]
        self.mu, self.sd = np.nanmean(smp, 1), np.nanstd(smp, 1) + 1e-6
        lab = np.full((H, W), -1, np.int16)
        lab[rows, cols] = ye
        rng = np.random.default_rng(0)
        pick = rng.choice(len(rows), min(len(rows), 2000), replace=False)
        Xp = np.zeros((len(pick), B, P, P), np.float32)
        Yp = np.full((len(pick), P, P), -1, np.int64)
        for n, i in enumerate(pick):
            r0 = int(np.clip(rows[i] - rng.integers(2, P - 2), 0, max(H - P, 0)))
            c0 = int(np.clip(cols[i] - rng.integers(2, P - 2), 0, max(W - P, 0)))
            h, w = min(P, H - r0), min(P, W - c0)
            Xp[n, :, :h, :w] = self._n(stack[:, r0:r0 + h, c0:c0 + w])
            Yp[n, :h, :w] = lab[r0:r0 + h, c0:c0 + w]
        K = len(self.classes_)
        self.net = UNetNet(B, K).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        lossf = nn.CrossEntropyLoss(weight=_class_weights(ye, K), ignore_index=-1)
        Xt, Yt = torch.tensor(Xp), torch.tensor(Yp)
        ep = self.p["epochs"]
        self.log(f"    U-Net: {len(pick)} patches of {P}x{P}, {ep} epochs on {dev}")
        for e in range(ep):
            self.net.train()
            perm, tot = torch.randperm(len(Xt)), 0.0
            for s in range(0, len(Xt), 16):
                b = perm[s:s + 16]
                if len(b) < 2:
                    continue
                xb, yb = Xt[b], Yt[b]
                if rng.random() < 0.5:
                    xb, yb = xb.flip(3), yb.flip(2)
                if rng.random() < 0.5:
                    xb, yb = xb.flip(2), yb.flip(1)
                opt.zero_grad()
                loss = lossf(self.net(xb.to(dev)), yb.to(dev))
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            if (e + 1) % 10 == 0 or e == 0:
                self.log(f"    epoch {e + 1}/{ep}  loss {tot / len(Xt):.4f}")

    def predict_map(self, stack, valid):
        B, H, W = stack.shape
        T, M, dev = 256, 16, _device()
        pred = np.zeros((H, W), np.int16)
        conf = np.zeros((H, W), np.float32)
        self.net.eval()
        with torch.no_grad():
            for r in range(0, H, T):
                for c in range(0, W, T):
                    if not valid[r:r + T, c:c + T].any():
                        continue
                    r0, r1, c0, c1 = max(r - M, 0), min(r + T + M, H), max(c - M, 0), min(c + T + M, W)
                    a = self._n(stack[:, r0:r1, c0:c1])
                    ph, pw = (-a.shape[1]) % 4, (-a.shape[2]) % 4
                    a = np.pad(a, ((0, 0), (0, ph), (0, pw)), mode="reflect" if min(a.shape[1:]) > 4 else "constant")
                    pr = torch.softmax(self.net(torch.tensor(a[None]).to(dev)), 1)[0].cpu().numpy()
                    ih, iw = min(T, H - r), min(T, W - c)
                    pr = pr[:, r - r0:r - r0 + ih, c - c0:c - c0 + iw]
                    pred[r:r + ih, c:c + iw] = self.classes_[pr.argmax(0)]
                    conf[r:r + ih, c:c + iw] = pr.max(0)
        return pred, conf


_REG = {"RF": RF, "XGB": XGB, "SVM": SVM, "DT": DT, "MLC": MLC, "CNN": CNN,
        "LSTM": LSTM, "UNET": UNET, "TRANS": TRANS}


def build_model(key, K, seq_shape, params, log):
    return _REG[key](K, seq_shape, params, log)
