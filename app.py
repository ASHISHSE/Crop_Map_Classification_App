"""Crop Map Generator - GUI (PyQt5).  Run:  python app.py
Developed by ASHISH SELOKAR  |  ashish111.selokar@gmail.com
"""
import os
import sys
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT as Toolbar
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import geopandas as gpd
from PyQt5 import QtCore, QtGui, QtWidgets as W

import core
import models
import unsup

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(HERE, "assets", "logo.png")
AUTHOR = "ASHISH SELOKAR"
EMAIL = "ashish111.selokar@gmail.com"
GREEN, TEAL, DARK = "#1b5e20", "#00796b", "#0f3d14"

STYLE = """
* { font-family: 'Segoe UI', Arial; font-size: 10pt; }
QMainWindow, QWidget#body { background: #eaf1ea; }
QScrollArea { border: none; background: transparent; }
QWidget#panel { background: transparent; }
QFrame#header { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0f3d14, stop:0.55 #1b5e20, stop:1 #00897b); }
QLabel#title { color: white; font-size: 21pt; font-weight: 700; }
QLabel#subtitle { color: #c8e6c9; font-size: 9pt; }
QLabel#logobadge { background: white; border-radius: 10px; padding: 3px 8px; border: 2px solid #a5d6a7; }
QLabel#badge { color: #e0f2f1; font-size: 9pt; }
QPushButton#about { background: rgba(255,255,255,0.15); color: white; border: 1px solid rgba(255,255,255,0.5);
                    border-radius: 14px; padding: 5px 16px; }
QPushButton#about:hover { background: rgba(255,255,255,0.3); }
QFrame#footer { background: #0f3d14; }
QLabel#footertxt { color: #c8e6c9; font-size: 9pt; }
QLabel#footertxt a { color: #80cbc4; text-decoration: none; }
QGroupBox { background: white; border: 1px solid #cfdccf; border-radius: 10px; margin-top: 16px;
            padding: 14px 10px 8px 10px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 2px 10px;
                   background: #2e7d32; color: white; border-radius: 9px; font-weight: 600; }
QLabel { color: #263238; }
QLabel#sect { color: #1b5e20; font-weight: 700; border-bottom: 1px solid #a5d6a7; padding: 2px 0; margin-top: 4px; }
QLabel#hint { color: #607d8b; font-size: 8.5pt; }
QPushButton { background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 7px; padding: 6px 12px; color: #1b5e20; }
QPushButton:hover { background: #c8e6c9; }
QPushButton:pressed { background: #a5d6a7; }
QPushButton:disabled { background: #eceff1; color: #90a4ae; border-color: #cfd8dc; }
QPushButton#prep { background: #00796b; color: white; border: none; font-weight: 600; padding: 8px; }
QPushButton#prep:hover { background: #00695c; }
QPushButton#run { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2e7d32, stop:1 #43a047); color: white;
                  border: none; font-size: 12pt; font-weight: 700; padding: 11px; border-radius: 9px; }
QPushButton#run:hover { background: #1b5e20; }
QPushButton#runu { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00695c, stop:1 #26a69a); color: white;
                   border: none; font-size: 11pt; font-weight: 700; padding: 10px; border-radius: 9px; }
QPushButton#runu:hover { background: #004d40; }
QPushButton#runu:disabled { background: #b0bec5; }
QPushButton#run:disabled { background: #b0bec5; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #fafdfa; border: 1px solid #b7c9b7; border-radius: 6px;
                                                padding: 4px 6px; min-height: 20px; }
QComboBox:focus, QSpinBox:focus, QLineEdit:focus, QDoubleSpinBox:focus { border: 1px solid #2e7d32; }
QCheckBox { spacing: 7px; color: #263238; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #81a581; border-radius: 4px; background: white; }
QCheckBox::indicator:checked { background: #2e7d32; border-color: #1b5e20; }
QTabWidget::pane { border: 1px solid #cfdccf; background: white; border-radius: 8px; top: -1px; }
QTabBar::tab { background: #d5e6d5; color: #2e5230; padding: 9px 20px; margin-right: 3px;
               border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 600; }
QTabBar::tab:selected { background: white; color: #1b5e20; border-bottom: 3px solid #2e7d32; }
QTabBar::tab:hover:!selected { background: #c1dcc1; }
QTableWidget { background: white; alternate-background-color: #f3f8f3; gridline-color: #dfe8df;
               border: 1px solid #dfe8df; selection-background-color: #c8e6c9; selection-color: #0f3d14; }
QHeaderView::section { background: #2e7d32; color: white; padding: 5px; border: none;
                       border-right: 1px solid #1b5e20; font-weight: 600; }
QPlainTextEdit { background: #10261a; color: #b9f6ca; font-family: Consolas, 'Courier New'; font-size: 9pt;
                 border-radius: 8px; padding: 6px; }
QProgressBar { border: none; background: #cfdccf; border-radius: 4px; max-height: 8px; }
QProgressBar::chunk { background: #43a047; border-radius: 4px; }
QLabel#card { background: white; border: 1px solid #cfdccf; border-radius: 10px; padding: 8px; }
QScrollBar:vertical { background: transparent; width: 10px; }
QScrollBar::handle:vertical { background: #a5c8a5; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QToolBar { background: #f3f8f3; border: none; }
"""


# ------------------------------------------------------------------ worker
class Worker(QtCore.QThread):
    log = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object)
    fail = QtCore.pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(self.log.emit))
        except Exception as e:
            self.log.emit(traceback.format_exc())
            self.fail.emit(str(e))


# ------------------------------------------------------------------ map helpers
def extent_of(img):
    t, H, Wd = img["transform"], img["H"], img["W"]
    return (t.c, t.c + Wd * t.a, t.f + H * t.e, t.f)


def background(img):
    st = img["stack"]
    s = max(1, max(img["H"], img["W"]) // 1200)
    a = np.nan_to_num(st[:min(3, st.shape[0]), ::s, ::s])
    out = np.zeros((a.shape[1], a.shape[2], 3), np.float32)
    for i in range(3):
        b = a[min(i, a.shape[0] - 1)]
        nz = b[b != 0]
        lo, hi = (np.percentile(nz, (2, 98)) if nz.size else (0, 1))
        out[..., i] = np.clip((b - lo) / (hi - lo + 1e-9), 0, 1)
    return out


def px_to_xy(img, rows, cols):
    t = img["transform"]
    return t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e


def hexcol(c):
    return "#%02x%02x%02x" % tuple(c)


def logo_array():
    """Logo as an image array for matplotlib (None if the file is missing)."""
    try:
        return matplotlib.image.imread(LOGO)
    except Exception:
        return None


def logo_pixmap(height):
    pm = QtGui.QPixmap(LOGO)
    if pm.isNull():
        return None
    return pm.scaledToHeight(height, QtCore.Qt.SmoothTransformation)


def add_logo(fig, x=0.775, y=0.004, h=0.062):
    """Small logo in the bottom-right corner of a map figure (next to the credit text)."""
    arr = logo_array()
    if arr is None:
        return
    fw, fh = fig.get_size_inches()
    w = h * fh / fw * arr.shape[1] / arr.shape[0]
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(arr)
    ax.axis("off")


def nice_len(w):
    raw = w / 5.0
    p = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * p:
            return m * p
    return p


def decorate_map(fig, ax, img, title, subtitle=""):
    """Coordinates grid, north arrow, scale bar, title and credit."""
    ax.set_facecolor("#f1f5f1")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=7, direction="in", colors="#37474f")
    for lab in ax.get_yticklabels():
        lab.set_rotation(90)
        lab.set_va("center")
    ax.grid(True, ls=":", lw=0.6, color="#8da08d", alpha=0.9)
    for sp in ax.spines.values():
        sp.set_color("#37474f")
        sp.set_linewidth(1.1)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=GREEN, loc="left", pad=16 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=7.5, color="#607d8b", va="bottom")
    ax.annotate("N", xy=(0.955, 0.965), xytext=(0.955, 0.865), xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=11, fontweight="bold", color="#263238",
                arrowprops=dict(arrowstyle="-|>", color="#263238", lw=1.6),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    x0, x1, y0, y1 = extent_of(img)
    if not img["crs"].is_geographic:
        L = nice_len(x1 - x0)
        lab = f"{L / 1000:g} km" if L >= 1000 else f"{L:g} m"
        ax.add_artist(AnchoredSizeBar(ax.transData, L, lab, "lower left", pad=0.5, color="#263238", frameon=True,
                                      size_vertical=(y1 - y0) * 0.006, fontproperties={"size": 8}))
    fig.text(0.995, 0.004, f"Developed by {AUTHOR}", ha="right", va="bottom", fontsize=6, color="#78909c")
    add_logo(fig)


def make_table(headers, rows, row_headers=None, colorfn=None):
    t = W.QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    if row_headers:
        t.setVerticalHeaderLabels(row_headers)
    else:
        t.verticalHeader().setVisible(False)
    t.setAlternatingRowColors(True)
    t.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            it = W.QTableWidgetItem(str(v))
            it.setTextAlignment(QtCore.Qt.AlignCenter if j or row_headers else QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            if colorfn:
                c = colorfn(i, j)
                if c is not None:
                    it.setBackground(QtGui.QBrush(c))
            t.setItem(i, j, it)
    t.resizeColumnsToContents()
    t.horizontalHeader().setStretchLastSection(True)
    return t


def rdylgn(v):
    r, g, b, _ = matplotlib.colormaps["RdYlGn"](0.15 + 0.7 * float(v))
    return QtGui.QColor(int(r * 255), int(g * 255), int(b * 255), 170)


def card(title, value, sub="", color=GREEN):
    l = W.QLabel(f"<div style='color:#607d8b;font-size:8.5pt'>{title}</div>"
                 f"<div style='color:{color};font-size:20pt;font-weight:700'>{value}</div>"
                 f"<div style='color:#78909c;font-size:8pt'>{sub}</div>")
    l.setObjectName("card")
    l.setAlignment(QtCore.Qt.AlignCenter)
    return l


# ------------------------------------------------------------------ result tab
class ResultTab(W.QWidget):
    def __init__(self, res, names, colors, img, ctx, title):
        super().__init__()
        self.res, self.names, self.colors, self.img, self.ctx, self.title = res, names, colors, img, ctx, title
        lay = W.QHBoxLayout(self)
        left = W.QVBoxLayout()
        self.fig = Figure(figsize=(8, 5.5), facecolor="white")
        self.cv = Canvas(self.fig)
        left.addWidget(Toolbar(self.cv, self))
        left.addWidget(self.cv, 1)
        row = W.QHBoxLayout()
        self.chk = W.QCheckBox("Show test points  (green = correct, red = wrong)")
        self.chk.toggled.connect(self.draw)
        b = W.QPushButton("Save map as PNG ...")
        b.clicked.connect(self.save_png)
        row.addWidget(self.chk)
        row.addStretch()
        row.addWidget(b)
        left.addLayout(row)
        lay.addLayout(left, 5)

        right = W.QVBoxLayout()
        a, at = res["acc"], res["acc_train"]
        cards = W.QHBoxLayout()
        cards.addWidget(card("OVERALL ACCURACY (test)", f"{a['oa'] * 100:.1f}%", f"train {at['oa'] * 100:.1f}%"))
        cards.addWidget(card("KAPPA (test)", f"{a['kappa']:.3f}", f"train {at['kappa']:.3f}", TEAL))
        cards.addWidget(card("TEST SAMPLES", f"{a['n']:,}", f"train {at['n']:,}", "#455a64"))
        right.addLayout(cards)
        sub = W.QTabWidget()
        sub.addTab(self.matrix_table(a), "Error matrix")
        sub.addTab(self.class_table(a), "Producer / User acc.")
        sub.addTab(self.area_table(), "Area (ha)")
        right.addWidget(sub, 1)
        lay.addLayout(right, 4)
        self.draw()

    def matrix_table(self, a):
        n, cm = len(self.names), a["cm"]
        rs = cm.sum(1)
        rows = []
        for i in range(n):
            rows.append([cm[i, j] for j in range(n)] + [cm[i].sum(),
                        "-" if np.isnan(a["pa"][i]) else f"{a['pa'][i] * 100:.1f}"])
        rows.append([cm[:, j].sum() for j in range(n)] + [cm.sum(), ""])
        rows.append(["-" if np.isnan(a["ua"][j]) else f"{a['ua'][j] * 100:.1f}" for j in range(n)] + ["", f"OA {a['oa'] * 100:.1f}"])

        def cf(i, j):
            if i < n and j < n and rs[i] > 0:
                f = cm[i, j] / rs[i]
                if i == j:
                    return QtGui.QColor(76, 175, 80, int(60 + 150 * f))
                return QtGui.QColor(239, 83, 80, int(200 * f)) if cm[i, j] else None
            return QtGui.QColor("#e8f5e9") if (i >= n or j >= n) else None
        t = make_table(self.names + ["Total", "Producer %"], rows, self.names + ["Total", "User %"], cf)
        return t

    def class_table(self, a):
        rows = [[nm, "-" if np.isnan(p) else f"{p * 100:.1f}", "-" if np.isnan(u) else f"{u * 100:.1f}"]
                for nm, p, u in zip(self.names, a["pa"], a["ua"])]
        vals = [(a["pa"][i], a["ua"][i]) for i in range(len(self.names))]
        return make_table(["Class", "Producer acc. %", "User acc. %"], rows,
                          colorfn=lambda i, j: rdylgn(vals[i][j - 1]) if j and not np.isnan(vals[i][j - 1]) else None)

    def area_table(self):
        px = np.asarray(self.res.get("class_pixel_counts", [0] * len(self.names)), float)
        ha = np.asarray(self.res.get("class_areas_ha", [0] * len(self.names)), float)
        tot = max(ha.sum(), 1e-9)
        rows = [[n, f"{int(p):,}", f"{h:,.2f}", f"{100 * h / tot:.2f}"] for n, p, h in zip(self.names, px, ha)]
        rows.append(["TOTAL", f"{int(px.sum()):,}", f"{ha.sum():,.2f}", "100.00"])
        return make_table(["Class", "Pixels", "Area (ha)", "Area (%)"], rows)

    def draw(self):
        self.fig.clear()
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3.2, 1.25], wspace=0.28, left=0.07, right=0.98, top=0.9, bottom=0.07)
        ax, bx = self.fig.add_subplot(gs[0]), self.fig.add_subplot(gs[1])
        arr, K = self.res["map"], len(self.names)
        s = max(1, max(arr.shape) // 1500)
        m = np.ma.masked_equal(arr[::s, ::s], 0)
        cmap = ListedColormap([tuple(v / 255 for v in c) for c in self.colors])
        ax.imshow(m, cmap=cmap, vmin=0.5, vmax=K + 0.5, extent=extent_of(self.img), interpolation="nearest")
        if self.ctx is not None:
            self.ctx.boundary.plot(ax=ax, color="#455a64", linewidth=0.5)
        gpd.GeoSeries([self.img["region_geom"]], crs=self.img["crs"]).boundary.plot(ax=ax, color="black", linewidth=1.3)
        x0, x1, y0, y1 = extent_of(self.img)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        if self.chk.isChecked():
            r, c, ok = self.res["te_rows"], self.res["te_cols"], self.res["te_correct"]
            sel = np.random.default_rng(0).permutation(len(r))[:4000]
            x, y = px_to_xy(self.img, r[sel], c[sel])
            ax.scatter(x, y, c=np.where(ok[sel], "#00e676", "#ff1744"), s=9, edgecolors="k", linewidths=0.3, zorder=5)
            ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="#00e676", mec="k", label="Test: correct"),
                               Line2D([], [], marker="o", ls="", mfc="#ff1744", mec="k", label="Test: wrong")],
                      loc="upper left", fontsize=7, framealpha=0.9)
        a = self.res["acc"]
        decorate_map(self.fig, ax, self.img, f"{self.title}  -  {self.res['label']}",
                     f"Classified crop map  |  Overall accuracy {a['oa'] * 100:.1f}%   Kappa {a['kappa']:.3f}")
        ha = np.asarray(self.res.get("class_areas_ha", [0] * K), float)
        ypos = np.arange(K)[::-1]
        bx.barh(ypos, ha, color=[hexcol(c) for c in self.colors], edgecolor="#37474f", linewidth=0.5)
        bx.set_yticks(ypos)
        bx.set_yticklabels(self.names, fontsize=7.5)
        for y, h in zip(ypos, ha):
            bx.text(h, y, f" {h:,.0f}", va="center", fontsize=7, color="#263238")
        bx.set_xlim(0, max(ha.max() * 1.35, 1))
        bx.set_title("Area by class (ha)", fontsize=9.5, fontweight="bold", color=GREEN, loc="left")
        bx.tick_params(axis="x", labelsize=7)
        for sp in ("top", "right"):
            bx.spines[sp].set_visible(False)
        bx.grid(axis="x", ls=":", alpha=0.6)
        self.cv.draw()

    def save_png(self):
        f, _ = W.QFileDialog.getSaveFileName(self, "Save map", f"{self.title}_{self.res['label']}.png".replace(" ", "_"),
                                             "PNG (*.png)")
        if f:
            self.fig.savefig(f, dpi=220, bbox_inches="tight")



# ------------------------------------------------------------------ unsupervised result tab
class UnsupResultTab(W.QWidget):
    def __init__(self, res, img, ctx, title):
        super().__init__()
        self.res, self.img, self.ctx, self.title = res, img, ctx, title
        self.names, self.colors = res["names"], res["colors"]
        lay = W.QHBoxLayout(self)
        left = W.QVBoxLayout()
        self.fig = Figure(figsize=(8, 5.5), facecolor="white")
        self.cv = Canvas(self.fig)
        left.addWidget(Toolbar(self.cv, self))
        left.addWidget(self.cv, 1)
        row = W.QHBoxLayout()
        row.addStretch()
        b = W.QPushButton("Save map as PNG ...")
        b.clicked.connect(self.save_png)
        row.addWidget(b)
        left.addLayout(row)
        lay.addLayout(left, 5)

        right = W.QVBoxLayout()
        ha = np.asarray(res["class_areas_ha"], float)
        cards = W.QHBoxLayout()
        cards.addWidget(card("CLASSES", f"{res['n_classes']}", res["label"]))
        cards.addWidget(card("ITERATIONS", f"{res['n_iter']}",
                             "converged" if res["converged"] else "max iterations reached", TEAL))
        cards.addWidget(card("MAPPED AREA", f"{ha.sum():,.0f}", "hectares", "#455a64"))
        right.addLayout(cards)
        sub = W.QTabWidget()
        sub.addTab(self.area_table(), "Area (ha)")
        sub.addTab(self.sig_table(), "Signatures (mean)")
        sub.addTab(self.profile(), "Profile")
        right.addWidget(sub, 1)
        lay.addLayout(right, 4)
        self.draw()

    def area_table(self):
        px = np.asarray(self.res["class_pixel_counts"], float)
        ha = np.asarray(self.res["class_areas_ha"], float)
        tot = max(ha.sum(), 1e-9)
        rows = [[n, f"{int(p):,}", f"{h:,.2f}", f"{100 * h / tot:.2f}"] for n, p, h in zip(self.names, px, ha)]
        rows.append(["TOTAL", f"{int(px.sum()):,}", f"{ha.sum():,.2f}", "100.00"])
        cols = self.colors
        return make_table(["Class", "Pixels", "Area (ha)", "Area (%)"], rows,
                          colorfn=lambda i, j: QtGui.QColor(*cols[i], 120) if (j == 0 and i < len(cols)) else None)

    def sig_table(self):
        m = self.res["signatures"]["Mean"]
        rows = [[idx] + [f"{v:.4f}" for v in r] for idx, r in zip(m.index, m.values)]
        cols = self.colors
        return make_table(["Class"] + list(m.columns), rows,
                          colorfn=lambda i, j: QtGui.QColor(*cols[i], 120) if j == 0 else None)

    def profile(self):
        f = Figure(figsize=(5, 3.6), facecolor="white")
        cv = Canvas(f)
        ax = f.add_subplot(111)
        m = self.res["signatures"]["Mean"]
        x = np.arange(m.shape[1])
        for i, (idx, r) in enumerate(m.iterrows()):
            ax.plot(x, r.values, marker="o", ms=3, lw=1.6, color=hexcol(self.colors[i]), label=idx)
        step = max(1, len(x) // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(list(m.columns)[::step], rotation=60, ha="right", fontsize=6.5)
        ax.set_ylabel("Mean pixel value", fontsize=8)
        ax.set_title("Class signature profile (mean per band)", fontsize=10, fontweight="bold", color=GREEN, loc="left")
        ax.grid(ls=":", alpha=0.6)
        ax.legend(fontsize=6.5, ncol=2 if len(m) > 8 else 1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        f.tight_layout()
        return cv

    def draw(self):
        self.fig.clear()
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3.2, 1.25], wspace=0.28, left=0.07, right=0.98, top=0.9, bottom=0.07)
        ax, bx = self.fig.add_subplot(gs[0]), self.fig.add_subplot(gs[1])
        arr, K = self.res["map"], len(self.names)
        s = max(1, max(arr.shape) // 1500)
        m = np.ma.masked_equal(arr[::s, ::s], 0)
        cmap = ListedColormap([tuple(v / 255 for v in c) for c in self.colors])
        ax.imshow(m, cmap=cmap, vmin=0.5, vmax=K + 0.5, extent=extent_of(self.img), interpolation="nearest")
        if self.ctx is not None:
            self.ctx.boundary.plot(ax=ax, color="#455a64", linewidth=0.5)
        gpd.GeoSeries([self.img["region_geom"]], crs=self.img["crs"]).boundary.plot(ax=ax, color="black", linewidth=1.3)
        x0, x1, y0, y1 = extent_of(self.img)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        decorate_map(self.fig, ax, self.img, f"{self.title}  -  {self.res['label']}",
                     f"Unsupervised classification  |  {K} classes  |  {self.res['n_iter']} iteration(s)")
        ha = np.asarray(self.res["class_areas_ha"], float)
        ypos = np.arange(K)[::-1]
        bx.barh(ypos, ha, color=[hexcol(c) for c in self.colors], edgecolor="#37474f", linewidth=0.5)
        bx.set_yticks(ypos)
        bx.set_yticklabels(self.names, fontsize=7.5)
        for y, h in zip(ypos, ha):
            bx.text(h, y, f" {h:,.0f}", va="center", fontsize=7)
        bx.set_xlim(0, max(ha.max() * 1.35, 1))
        bx.set_title("Area by class (ha)", fontsize=9.5, fontweight="bold", color=GREEN, loc="left")
        bx.tick_params(axis="x", labelsize=7)
        for sp in ("top", "right"):
            bx.spines[sp].set_visible(False)
        bx.grid(axis="x", ls=":", alpha=0.6)
        self.cv.draw()

    def save_png(self):
        f, _ = W.QFileDialog.getSaveFileName(self, "Save map", f"{self.title}_{self.res['label']}.png".replace(" ", "_"),
                                             "PNG (*.png)")
        if f:
            self.fig.savefig(f, dpi=220, bbox_inches="tight")

# ------------------------------------------------------------------ main window
class Main(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crop Map Generator  |  RF / ML / DL classification of satellite imagery")
        self.resize(1520, 940)
        if os.path.exists(LOGO):
            self.setWindowIcon(QtGui.QIcon(LOGO))
        self.paths, self.gt, self.img, self.img_key, self.prep, self.region = [], None, None, None, None, None
        self.bounds, self.ctx, self.worker = None, None, None
        self.build_ui()
        self.welcome()
        QtCore.QTimer.singleShot(200, self.load_bounds)

    # ---------------- UI
    def build_ui(self):
        root = W.QWidget()
        self.setCentralWidget(root)
        rv = W.QVBoxLayout(root)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        rv.addWidget(self.make_header())
        body = W.QWidget()
        body.setObjectName("body")
        lay = W.QHBoxLayout(body)
        lay.setContentsMargins(12, 10, 12, 10)
        rv.addWidget(body, 1)
        rv.addWidget(self.make_footer())

        panel = W.QWidget()
        panel.setObjectName("panel")
        pl = W.QVBoxLayout(panel)
        pl.setContentsMargins(2, 0, 8, 0)
        sc = W.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(panel)
        sc.setFixedWidth(440)
        lay.addWidget(sc)

        def group(title):
            g = W.QGroupBox(title)
            f = W.QFormLayout(g)
            f.setLabelAlignment(QtCore.Qt.AlignLeft)
            pl.addWidget(g)
            return f

        def hint(f, text):
            l = W.QLabel(text)
            l.setObjectName("hint")
            l.setWordWrap(True)
            f.addRow(l)
            return l

        f = group("1 | Input image stack")
        b = W.QPushButton("Add image(s) ...")
        b.clicked.connect(self.pick_images)
        c = W.QPushButton("Clear")
        c.clicked.connect(lambda: (self.paths.clear(), self.lbl_img.setText("No images selected")))
        row = W.QHBoxLayout()
        row.addWidget(b, 2)
        row.addWidget(c, 1)
        f.addRow(row)
        self.lbl_img = hint(f, "No images selected")
        hint(f, "FCC / NDVI stack - one multi-band file or several files.")

        f = group("2 | Study area (inbuilt boundaries)")
        self.cb_dist, self.cb_tal = W.QComboBox(), W.QComboBox()
        self.cb_dist.currentTextChanged.connect(self.dist_changed)
        f.addRow("District", self.cb_dist)
        f.addRow("Taluka (optional)", self.cb_tal)

        f = group("3 | Training data (GT)")
        b = W.QPushButton("Open GT shapefile (points / polygons) ...")
        b.clicked.connect(self.pick_gt)
        f.addRow(b)
        self.lbl_gt = hint(f, "No GT loaded")
        self.cb_col = W.QComboBox()
        f.addRow("Class label column", self.cb_col)
        b = W.QPushButton("Clip GT to area  -  check, map && count")
        b.setObjectName("prep")
        b.clicked.connect(self.prepare)
        f.addRow(b)

        f = group("4 | Sampling")
        self.sp_tr, self.sp_te = W.QSpinBox(), W.QSpinBox()
        for s in (self.sp_tr, self.sp_te):
            s.setRange(1, 99)
            s.setSuffix(" %")
        self.sp_tr.setValue(70)
        self.sp_te.setValue(30)
        self.sp_tr.valueChanged.connect(lambda v: self.sp_te.setValue(100 - v) if self.sp_te.value() != 100 - v else None)
        self.sp_te.valueChanged.connect(lambda v: self.sp_tr.setValue(100 - v) if self.sp_tr.value() != 100 - v else None)
        f.addRow("Training samples", self.sp_tr)
        f.addRow("Testing samples", self.sp_te)
        self.ck_group = W.QCheckBox("Split by polygon / point (no spatial leakage)")
        self.ck_group.setChecked(True)
        f.addRow(self.ck_group)
        self.sp_max = W.QSpinBox()
        self.sp_max.setRange(0, 1_000_000)
        self.sp_max.setValue(5000)
        self.sp_max.setSpecialValueText("no limit")
        f.addRow("Max train pixels / class", self.sp_max)

        f = group("5 | Classification methods")
        sect = W.QLabel("SUPERVISED  (needs GT - select up to 3)")
        sect.setObjectName("sect")
        f.addRow(sect)
        self.method_cb = {}
        grid = W.QGridLayout()
        for n, (k, lab) in enumerate(models.METHODS.items()):
            ck = W.QCheckBox(lab)
            ck.toggled.connect(self.limit_methods)
            self.method_cb[k] = ck
            grid.addWidget(ck, n // 2, n % 2)
        f.addRow(grid)
        sect = W.QLabel("UNSUPERVISED  (no GT needed)")
        sect.setObjectName("sect")
        f.addRow(sect)
        self.ck_km = W.QCheckBox("K-Means  (default)")
        self.ck_km.setChecked(True)
        self.ck_iso = W.QCheckBox("ISODATA")
        row = W.QHBoxLayout()
        row.addWidget(self.ck_km)
        row.addWidget(self.ck_iso)
        f.addRow(row)

        f = group("Parameters")
        sect = W.QLabel("Supervised")
        sect.setObjectName("sect")
        f.addRow(sect)
        self.sp_trees = W.QSpinBox()
        self.sp_trees.setRange(10, 2000)
        self.sp_trees.setValue(200)
        self.sp_ep = W.QSpinBox()
        self.sp_ep.setRange(1, 1000)
        self.sp_ep.setValue(40)
        self.sp_conf = W.QDoubleSpinBox()
        self.sp_conf.setRange(0, 1)
        self.sp_conf.setSingleStep(0.05)
        self.sp_conf.setValue(0.40)
        self.sp_conf.setToolTip("Pixels whose best-class probability is below this go to 'Other crop'. 0 = off.")
        f.addRow("Trees (RF / XGB)", self.sp_trees)
        f.addRow("Epochs (DL)", self.sp_ep)
        f.addRow("Min. confidence (else Other)", self.sp_conf)

        # ---- K-Means parameters (shown when K-Means is ticked)
        self.km_box = W.QWidget()
        kf = W.QFormLayout(self.km_box)
        kf.setContentsMargins(0, 0, 0, 0)
        sect = W.QLabel("Unsupervised - K-Means")
        sect.setObjectName("sect")
        kf.addRow(sect)
        self.sp_km_k = W.QSpinBox()
        self.sp_km_k.setRange(2, 60)
        self.sp_km_k.setValue(unsup.KMEANS_DEFAULTS["n_classes"])
        self.sp_km_it = W.QSpinBox()
        self.sp_km_it.setRange(1, 1000)
        self.sp_km_it.setValue(unsup.KMEANS_DEFAULTS["iterations"])
        kf.addRow("No. of classes", self.sp_km_k)
        kf.addRow("Iterations (max)", self.sp_km_it)
        f.addRow(self.km_box)
        self.ck_km.toggled.connect(self.km_box.setVisible)

        # ---- ISODATA parameters (shown when ISODATA is ticked)
        self.iso_box = W.QWidget()
        isf = W.QFormLayout(self.iso_box)
        isf.setContentsMargins(0, 0, 0, 0)
        sect = W.QLabel("Unsupervised - ISODATA")
        sect.setObjectName("sect")
        isf.addRow(sect)
        spec = [("init_clusters", "Initial clusters", "i", 2, 60, 1),
                ("min_clusters", "Min. clusters", "i", 2, 60, 1),
                ("max_clusters", "Max. clusters", "i", 2, 100, 1),
                ("iterations", "Iterations (max)", "i", 1, 1000, 1),
                ("min_samples_pct", "Min. samples per cluster (%)", "d", 0.01, 20, 0.25),
                ("max_std", "Max. std. deviation (split)", "d", 0.05, 10, 0.1),
                ("min_dist", "Min. distance between means (merge)", "d", 0.01, 10, 0.1),
                ("max_merge_pairs", "Max. merge pairs", "i", 1, 20, 1),
                ("change_pct", "Change threshold (%)", "d", 0.1, 50, 0.5)]
        self.iso_w = {}
        for key, lab, kind, lo, hi, st in spec:
            w_ = W.QSpinBox() if kind == "i" else W.QDoubleSpinBox()
            w_.setRange(lo, hi)
            w_.setSingleStep(st)
            self.iso_w[key] = w_
            isf.addRow(lab, w_)
        self.reset_iso()
        b_ = W.QPushButton("Reset ISODATA to default parameters")
        b_.clicked.connect(self.reset_iso)
        isf.addRow(b_)
        f.addRow(self.iso_box)
        self.ck_iso.toggled.connect(self.iso_box.setVisible)
        self.iso_box.setVisible(False)

        self.ck_sig = W.QCheckBox("Extract class signature values to Excel (unsupervised)")
        self.ck_sig.setChecked(True)
        f.addRow(self.ck_sig)

        f = group("6 | Output")
        self.ed_out = W.QLineEdit()
        self.ed_out.setPlaceholderText("Output folder ...")
        b = W.QPushButton("Browse")
        b.clicked.connect(lambda: self.ed_out.setText(W.QFileDialog.getExistingDirectory(self, "Output folder") or self.ed_out.text()))
        row = W.QHBoxLayout()
        row.addWidget(self.ed_out, 1)
        row.addWidget(b)
        f.addRow(row)
        self.btn_run = W.QPushButton("RUN CLASSIFICATION")
        self.btn_run.setObjectName("run")
        self.btn_run.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self.run)
        pl.addWidget(self.btn_run)
        self.pb = W.QProgressBar()
        self.pb.setRange(0, 1)
        self.pb.setTextVisible(False)
        pl.addWidget(self.pb)
        pl.addStretch()

        self.tabs = W.QTabWidget()
        lay.addWidget(self.tabs, 1)
        w = W.QWidget()
        v = W.QVBoxLayout(w)
        self.fig = Figure(figsize=(9, 6), facecolor="white")
        self.cv = Canvas(self.fig)
        v.addWidget(Toolbar(self.cv, self))
        v.addWidget(self.cv)
        self.tabs.addTab(w, "Training data map")
        self.tbl = W.QTableWidget()
        self.tabs.addTab(self.tbl, "Class-wise counts")
        self.txt = W.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.tabs.addTab(self.txt, "Log")
        self.res_tabs = W.QTabWidget()
        self.tabs.addTab(self.res_tabs, "Results")

    def make_header(self):
        h = W.QFrame()
        h.setObjectName("header")
        h.setFixedHeight(92)
        l = W.QHBoxLayout(h)
        l.setContentsMargins(22, 6, 22, 6)
        t = W.QVBoxLayout()
        a = W.QLabel("Crop Map Generator")
        a.setObjectName("title")
        b = W.QLabel("Supervised: RF - XGBoost - SVM - DT - MLC - CNN - LSTM - U-Net - Transformer   |   Unsupervised: K-Means - ISODATA")
        b.setObjectName("subtitle")
        t.addWidget(a)
        t.addWidget(b)
        l.addLayout(t, 1)
        pm = logo_pixmap(62)
        if pm is not None:
            lg = W.QLabel()
            lg.setPixmap(pm)
            lg.setObjectName("logobadge")
            lg.setAlignment(QtCore.Qt.AlignCenter)
            l.addWidget(lg)
            l.addSpacing(10)
        d = W.QLabel(f"Developed by<br><b style='font-size:12pt;color:white'>{AUTHOR}</b>"
                     f"<br><span style='font-size:8pt;color:#b2dfdb'>{EMAIL}</span>")
        d.setObjectName("badge")
        d.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        l.addWidget(d)
        l.addSpacing(18)
        ab = W.QPushButton("About")
        ab.setObjectName("about")
        ab.clicked.connect(self.about)
        l.addWidget(ab)
        return h

    def make_footer(self):
        f = W.QFrame()
        f.setObjectName("footer")
        f.setFixedHeight(38)
        l = W.QHBoxLayout(f)
        l.setContentsMargins(18, 0, 18, 0)
        pm = logo_pixmap(26)
        if pm is not None:
            lg = W.QLabel()
            lg.setPixmap(pm)
            lg.setObjectName("logobadge")
            lg.setStyleSheet("background:white;border-radius:5px;padding:1px 4px;")
            l.addWidget(lg)
            l.addSpacing(8)
        t = W.QLabel(f"Developed by <b>{AUTHOR}</b> &nbsp;|&nbsp; Contact: "
                     f"<a href='mailto:{EMAIL}'><span style='color:#80cbc4;'>{EMAIL}</span></a>")
        t.setObjectName("footertxt")
        t.setTextFormat(QtCore.Qt.RichText)
        t.setOpenExternalLinks(True)
        pal = t.palette()
        pal.setColor(QtGui.QPalette.Link, QtGui.QColor("#80cbc4"))
        pal.setColor(QtGui.QPalette.LinkVisited, QtGui.QColor("#80cbc4"))
        t.setPalette(pal)
        l.addWidget(t)
        l.addStretch()
        v = W.QLabel("Crop Map Generator v3.0")
        v.setObjectName("footertxt")
        l.addWidget(v)
        return f

    def about(self):
        m = W.QMessageBox(self)
        m.setWindowTitle("About")
        m.setTextFormat(QtCore.Qt.RichText)
        pm = logo_pixmap(130)
        if pm is not None:
            m.setIconPixmap(pm)
        m.setText(f"<h3 style='color:{GREEN}'>Crop Map Generator v1.0</h3>"
                  "Crop classification of multi-image satellite stacks (FCC / NDVI) with "
                  "RF, XGBoost, SVM, DT, MLC, CNN, LSTM, U-Net and Transformer, "
                  "with accuracy assessment and class-wise area (ha).<br><br>"
                  f"<b>Developed by {AUTHOR}</b><br>Contact: <a href='mailto:{EMAIL}'>{EMAIL}</a>")
        m.exec_()

    def welcome(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        arr = logo_array()
        if arr is not None:
            ia = ax.inset_axes([0.38, 0.62, 0.24, 0.34])
            ia.imshow(arr)
            ia.axis("off")
        ax.text(0.5, 0.55, "Crop Map Generator", ha="center", fontsize=24, fontweight="bold", color=GREEN)
        ax.text(0.5, 0.45, "1  Add image(s)   2  Choose district / taluka   3  Open GT shapefile\n"
                           "4  Click 'Clip GT to area'   5  Pick 3 methods   6  Run",
                ha="center", va="top", fontsize=11, color="#546e7a", linespacing=1.9)
        ax.text(0.5, 0.06, f"Developed by {AUTHOR}  |  {EMAIL}", ha="center", fontsize=9, color="#78909c")
        self.cv.draw()

    # ---------------- helpers
    def log(self, s):
        self.txt.appendPlainText(s)

    def err(self, s):
        W.QMessageBox.warning(self, "Crop Map Generator", s)

    def limit_methods(self):
        sel = [k for k, c in self.method_cb.items() if c.isChecked()]
        if len(sel) > 3:
            self.sender().setChecked(False)
            self.err("Select at most 3 supervised methods.")

    def busy(self, on, msg=""):
        self.pb.setRange(0, 0 if on else 1)
        self.btn_run.setEnabled(not on)
        self.statusBar().showMessage(msg if on else "Ready")

    def start(self, fn, on_done, msg="Working ..."):
        self.busy(True, msg)
        self.worker = Worker(fn)
        self.worker.log.connect(self.log)
        self.worker.done.connect(lambda r: (self.busy(False), on_done(r)))
        self.worker.fail.connect(lambda m: (self.busy(False), self.err(m)))
        self.worker.start()

    # ---------------- inputs
    def load_bounds(self):
        try:
            self.bounds = core.load_boundaries()
            self.cb_dist.addItems(core.district_names(self.bounds))
            self.log(f"Boundaries loaded ({len(self.bounds['district'])} districts, "
                     f"{len(self.bounds['taluka'])} talukas).")
        except Exception as e:
            self.err(f"Could not load boundary shapefiles (check config.py):\n{e}")

    def dist_changed(self, d):
        self.cb_tal.clear()
        self.cb_tal.addItem("(whole district)")
        if self.bounds and d:
            self.cb_tal.addItems(core.taluka_names(self.bounds, d))

    def pick_images(self):
        fs, _ = W.QFileDialog.getOpenFileNames(self, "Select image(s)", "", "Rasters (*.tif *.tiff *.img *.vrt)")
        if fs:
            self.paths += [f for f in fs if f not in self.paths]
            self.lbl_img.setText("\n".join(os.path.basename(p) for p in self.paths))

    def pick_gt(self):
        f, _ = W.QFileDialog.getOpenFileName(self, "GT shapefile", "", "Vector (*.shp *.gpkg *.geojson)")
        if not f:
            return
        try:
            self.gt = core.load_gt(f)
        except Exception as e:
            return self.err(str(e))
        self.lbl_gt.setText(f"{os.path.basename(f)}: {len(self.gt)} features, {self.gt.geom_type.unique().tolist()}")
        self.cb_col.clear()
        self.cb_col.addItems([c for c in self.gt.columns if c != "geometry"])

    # ---------------- prepare
    def region_args(self):
        t = self.cb_tal.currentText()
        return self.cb_dist.currentText(), (None if t.startswith("(") else t)

    def prepare(self):
        if not self.paths:
            return self.err("Add at least one image.")
        if self.gt is None or not self.cb_col.currentText():
            return self.err("Open a GT shapefile and choose the class column.")
        if self.bounds is None:
            return self.err("Boundaries not loaded.")
        d, t = self.region_args()
        key = (tuple(self.paths), d, t)
        col, gt, paths, bounds, old = self.cb_col.currentText(), self.gt, list(self.paths), self.bounds, (self.img_key, self.img)

        def job(log):
            region = core.select_region(bounds, d, t)
            img = old[1] if old[0] == key else core.load_stack(paths, region, log)
            prep = core.prepare_training(gt, col, img, log)
            return region, img, prep

        def done(r):
            self.region, self.img, self.prep = r
            self.img_key = key
            self.update_ctx(d)
            self.show_training()
            self.tabs.setCurrentIndex(0)

        self.log(f"=== Preparing: {d} / {t or 'whole district'} ===")
        self.start(job, done, "Reading image and clipping GT ...")

    def area_title(self):
        d, t = self.region_args()
        return f"{d}" + (f" - {t}" if t else "")

    def show_training(self):
        img, prep = self.img, self.prep
        cls, gt, cn = prep["classes"], prep["gt"], prep["counts"]
        cols = core.class_colors(len(cls))
        self.fig.clear()
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3.2, 1.3], wspace=0.3, left=0.07, right=0.98, top=0.9, bottom=0.07)
        ax, bx = self.fig.add_subplot(gs[0]), self.fig.add_subplot(gs[1])
        ax.imshow(background(img), extent=extent_of(img), interpolation="bilinear")
        if self.ctx is not None and len(self.ctx):
            self.ctx.boundary.plot(ax=ax, color="#ffe082", linewidth=0.6)
            if len(self.ctx) <= 25:
                for _, r in self.ctx.iterrows():
                    p = r.geometry.representative_point()
                    ax.text(p.x, p.y, str(r[self.bounds["tcol"]]), fontsize=6, color="white", ha="center",
                            alpha=0.85, path_effects=_halo())
        gpd.GeoSeries([img["region_geom"]], crs=img["crs"]).boundary.plot(ax=ax, color="#ffeb3b", linewidth=2)
        for i, c in enumerate(cls):
            g = gt[gt["_cls"] == c]
            col = hexcol(cols[i])
            pts, pol = g[g.geom_type == "Point"], g[g.geom_type != "Point"]
            if len(pts):
                pts.plot(ax=ax, color=col, markersize=26, edgecolor="white", linewidth=0.7, zorder=4)
            if len(pol):
                pol.plot(ax=ax, facecolor=col, edgecolor="white", linewidth=0.6, alpha=0.9, zorder=3)
        rm = prep["removed"]
        handles = [Line2D([0], [0], color="#ffeb3b", lw=2, label="Selected area")]
        if len(rm):
            bad = gt[gt["_fid"].isin(rm["ID"])]
            if len(bad):
                rp = bad.geometry.representative_point()
                ax.scatter(rp.x, rp.y, marker="X", c="red", s=60, edgecolors="white", linewidths=0.8, zorder=6)
            handles.append(Line2D([], [], marker="X", ls="", mfc="red", mec="white", ms=8, label="Removed (not on image)"))
        ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9)
        x0, x1, y0, y1 = extent_of(img)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        decorate_map(self.fig, ax, img, f"Training data  -  {self.area_title()}",
                     f"{len(cls)} classes  |  {int(cn['Pixels'].sum()):,} training pixels  |  {len(rm)} feature(s) removed")
        ypos = np.arange(len(cls))[::-1]
        bx.barh(ypos, cn["Pixels"], color=[hexcol(c) for c in cols[:len(cls)]], edgecolor="#37474f", linewidth=0.5)
        bx.set_yticks(ypos)
        bx.set_yticklabels(cls, fontsize=7.5)
        for y, v in zip(ypos, cn["Pixels"]):
            bx.text(v, y, f" {int(v):,}", va="center", fontsize=7)
        bx.set_xlim(0, max(cn["Pixels"].max() * 1.35, 1))
        bx.set_title("Pixels per class", fontsize=9.5, fontweight="bold", color=GREEN, loc="left")
        bx.tick_params(axis="x", labelsize=7)
        for sp in ("top", "right"):
            bx.spines[sp].set_visible(False)
        bx.grid(axis="x", ls=":", alpha=0.6)
        self.cv.draw()

        rows = [[r.Class, r.Features, f"{r.Pixels:,}"] for r in cn.itertuples()]
        rows.append(["TOTAL", int(cn["Features"].sum()), f"{int(cn['Pixels'].sum()):,}"])
        t = make_table(["Class", "Features (points / polygons)", "Pixels"], rows)
        self.tabs.removeTab(1)
        self.tbl = t
        self.tabs.insertTab(1, t, "Class-wise counts")

    def update_ctx(self, d):
        try:
            self.ctx = core._talukas_in_district(self.bounds, d).to_crs(self.img["crs"])
        except Exception:
            self.ctx = None

    def reset_iso(self):
        for k, v in unsup.ISODATA_DEFAULTS.items():
            self.iso_w[k].setValue(v)

    # ---------------- run (supervised and/or unsupervised, same button)
    def run(self):
        sup = [(k, models.METHODS[k]) for k, c in self.method_cb.items() if c.isChecked()]
        uns = []
        if self.ck_km.isChecked():
            uns.append(("KMEANS", f"K-Means ({self.sp_km_k.value()} classes)"))
        if self.ck_iso.isChecked():
            uns.append(("ISODATA", "ISODATA"))
        if not sup and not uns:
            return self.err("Select at least one supervised or unsupervised method.")
        if not self.paths:
            return self.err("Add at least one image.")
        if self.bounds is None:
            return self.err("Boundaries not loaded.")
        out = self.ed_out.text().strip()
        if not out:
            return self.err("Choose an output folder.")
        d, t = self.region_args()
        key = (tuple(self.paths), d, t)
        if sup:
            if self.prep is None:
                return self.err("Supervised methods need GT: click 'Clip GT to area' first.")
            if key != self.img_key:
                return self.err("Images / district / taluka changed since the GT was prepared. "
                                "Click 'Clip GT to area' again.")
            if self.sp_tr.value() + self.sp_te.value() != 100:
                return self.err("Training % + Testing % must equal 100.")
        iso = {k: (int(w_.value()) if isinstance(w_, W.QSpinBox) else float(w_.value())) for k, w_ in self.iso_w.items()}
        if self.ck_iso.isChecked() and not (iso["min_clusters"] <= iso["init_clusters"] <= iso["max_clusters"]):
            return self.err("ISODATA: Min. clusters <= Initial clusters <= Max. clusters is required.")
        tag = (d + ("_" + t if t else "")).replace(" ", "_")
        p_sup = dict(train_pct=self.sp_tr.value(), by_group=self.ck_group.isChecked(),
                     max_per_class=self.sp_max.value(), seed=42, n_trees=self.sp_trees.value(),
                     epochs=self.sp_ep.value(), min_conf=self.sp_conf.value(), max_svm=8000)
        p_uns = dict(kmeans=dict(n_classes=self.sp_km_k.value(), iterations=self.sp_km_it.value()),
                     isodata=iso, extract_sig=self.ck_sig.isChecked(), seed=42)
        paths, bounds, prep = list(self.paths), self.bounds, self.prep
        old = (self.img_key, self.img, self.region)

        def job(log):
            if old[0] == key and old[1] is not None:
                region, img = old[2], old[1]
            else:
                region = core.select_region(bounds, d, t)
                img = core.load_stack(paths, region, log)
            r_sup = core.run_classification(img, prep, sup, p_sup, out, tag, log) if sup else None
            r_uns = core.run_unsupervised(img, uns, p_uns, out, tag, log) if uns else None
            return dict(region=region, img=img, sup=r_sup, uns=r_uns)

        def done(r):
            if key != self.img_key:
                self.prep = None                     # GT preparation belonged to another image/area
            self.region, self.img, self.img_key = r["region"], r["img"], key
            self.update_ctx(d)
            self.show_results(r)

        self.log(f"=== Classification | {d} / {t or 'whole district'} | "
                 f"supervised: {[l for _, l in sup]} | unsupervised: {[l for _, l in uns]} ===")
        self.tabs.setCurrentIndex(2)
        self.start(job, done, "Training / clustering and classifying ...")

    def show_results(self, r):
        self.res_tabs.clear()
        title = self.area_title()
        files = []
        if r["sup"]:
            o = r["sup"]
            for k, res in o["results"].items():
                self.res_tabs.addTab(ResultTab(res, o["names"], o["colors"], self.img, self.ctx, title), res["label"])
            files.append(f"Supervised Excel: {o['excel'] or 'NOT saved (see Log)'}")
        if r["uns"]:
            o = r["uns"]
            for k, res in o["results"].items():
                self.res_tabs.addTab(UnsupResultTab(res, self.img, self.ctx, title), res["label"])
            files.append(f"Unsupervised Excel: {o['excel'] or 'NOT saved (see Log)'}")
        self.tabs.setCurrentIndex(3)
        W.QMessageBox.information(self, "Done", "Finished.\n" + "\n".join(files))


def _halo():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=1.5, foreground="black")]


if __name__ == "__main__":
    app = W.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    w = Main()
    w.show()
    sys.exit(app.exec_())
