#!/usr/bin/env python3
"""
AquaSpectra - DIY Spectrometer v3
Camera capture + wavelength calibration + absorbance + MongoDB + ML identification.

Live-view keys:
    r  select ROI (drag a box over the spectrum band, press ENTER)
    c  calibrate wavelength axis (click known spectral lines)
    d  capture DARK frame (block all light)
    f  capture REFERENCE (light through distilled water / blank, no sample)
    s  capture spectrum, preview, predict (if a model exists), optionally save
    q  quit

CLI commands:
    python spectrometer.py live [--camera 1] [--avg 10] [--lock-camera --exposure -6 --gain 0]
    python spectrometer.py list [--sample NAME] [--label LABEL]
    python spectrometer.py plot ID [--signal absorbance]
    python spectrometer.py compare ID1 ID2 ... [--signal absorbance]
    python spectrometer.py export ID [-o file.csv]
    python spectrometer.py delete ID
    python spectrometer.py annotate ID --label salty --impurity NaCl --concentration 500
    python spectrometer.py import PATH [--label NAME]       # CSV file or folder of CSVs
    python spectrometer.py export-dataset -o dataset.csv    # or .npz, for external ML work
    python spectrometer.py match ID [--top 5]               # nearest stored spectra
    python spectrometer.py evaluate [--model rf|svm]        # cross-validation only
    python spectrometer.py train [--model rf|svm]           # train + save a model
    python spectrometer.py predict ID                       # analyse a stored spectrum

Config (environment variables):
    MONGO_URI   default mongodb://localhost:27017
    MONGO_DB    default spectrometer
    MODEL_DIR   default ./models

Notes:
  * Absorbance A = -log10((S - D) / (R - D)) needs a reference (R) and ideally a dark (D)
    captured with the SAME ROI x-range, camera settings and light as the sample.
  * Webcam output is gamma-encoded, so absorbance is approximate (fine for comparing
    samples measured on the same rig, not for absolute chemistry).
  * Models are saved with joblib (pickle). Only load model files you created yourself.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from bson import ObjectId
from pymongo import DESCENDING, MongoClient
from pymongo.errors import PyMongoError
from scipy.signal import find_peaks, savgol_filter

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    ML_OK = True
except ImportError:  # capture/storage still work without scikit-learn
    ML_OK = False

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "spectrometer")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

# ---- preprocessing / ML constants ------------------------------------------------
GRID = np.arange(400.0, 701.0, 2.0)   # common wavelength grid (nm) for all spectra
MIN_COVERAGE = 0.5                    # spectrum must cover >=50% of GRID to be usable
SMOOTH_WINDOW = 11                    # Savitzky-Golay window (odd)
BASELINE_ORDER = 1                    # gentle (linear) so broad absorption bands survive
AMP_FLOOR = 0.2                       # absorbance norms below this are treated as "noise"
MIN_REF_COUNTS = 5.0                  # ignore pixels where (reference - dark) is this dark
NOVELTY_FLOOR = 0.08                  # minimum novelty distance threshold
MIN_CONF = 0.5                        # below this the verdict is "UNCERTAIN"
SATURATION = 250


# ==================================================================================
# Database
# ==================================================================================
def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")  # fail fast if the server is unreachable
    db = client[MONGO_DB]
    db.spectra.create_index([("created_at", DESCENDING)])
    db.spectra.create_index("sample")
    db.spectra.create_index("label")
    db.calibrations.create_index([("active", 1), ("created_at", DESCENDING)])
    db.references.create_index([("kind", 1), ("active", 1), ("created_at", DESCENDING)])
    return db


def try_get_db():
    try:
        return get_db()
    except PyMongoError as e:
        print(f"[warn] MongoDB unavailable ({e.__class__.__name__}). Running without storage.")
        return None


def load_active_calibration(db):
    if db is None:
        return None
    return db.calibrations.find_one({"active": True}, sort=[("created_at", DESCENDING)])


def oid(s):
    try:
        return ObjectId(s)
    except Exception:
        sys.exit(f"Invalid id: {s}")


# ==================================================================================
# Camera
# ==================================================================================
def camera_settings(cap):
    props = {
        "width": cv2.CAP_PROP_FRAME_WIDTH, "height": cv2.CAP_PROP_FRAME_HEIGHT,
        "exposure": cv2.CAP_PROP_EXPOSURE, "gain": cv2.CAP_PROP_GAIN,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE, "auto_wb": cv2.CAP_PROP_AUTO_WB,
        "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE, "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST, "saturation": cv2.CAP_PROP_SATURATION,
    }
    return {k: float(cap.get(p)) for k, p in props.items()}


def lock_camera(cap, exposure=None, gain=None, wb_temp=None):
    """Turn off auto white-balance / auto-exposure so spectra are comparable between days.
    Driver support varies (0.25 = 'manual' on V4L2/most UVC; exposure units differ per
    OS/driver, e.g. negative log2 seconds on Windows). Check the read-back values."""
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    if exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, gain)
    if wb_temp is not None:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temp)
    time.sleep(0.5)
    for _ in range(5):  # flush frames captured with the old settings
        cap.read()
    return camera_settings(cap)


# ==================================================================================
# Spectrum processing
# ==================================================================================
def rgb_to_wavelength(r, g, b):
    """Legacy hue-based estimate from the first project (kept for reference only)."""
    hsv = cv2.cvtColor(np.uint8([[[int(r), int(g), int(b)]]]), cv2.COLOR_RGB2HSV)[0][0]
    return round(750 - (hsv[0] / 179.0) * (750 - 380), 1)


def capture_averaged(cap, roi, n_frames):
    """Average n frames of the ROI to reduce sensor noise."""
    x, y, w, h = roi
    acc, count = None, 0
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        crop = frame[y:y + h, x:x + w].astype(np.float64)
        acc = crop if acc is None else acc + crop
        count += 1
    if count == 0:
        return None
    return (acc / count).astype(np.float32)


def extract_spectrum(cropped):
    """Column-wise mean of each channel (cropped is BGR)."""
    b = cropped[:, :, 0].mean(axis=0)
    g = cropped[:, :, 1].mean(axis=0)
    r = cropped[:, :, 2].mean(axis=0)
    return r, g, b, (r + g + b) / 3.0


def fit_calibration(pixels, wavelengths):
    degree = 1 if len(pixels) == 2 else 2
    coeffs = np.polyfit(pixels, wavelengths, degree)
    residual = np.polyval(coeffs, pixels) - np.array(wavelengths)
    return coeffs.tolist(), degree, float(np.sqrt(np.mean(residual ** 2)))


def pixel_axis(roi_x, width):
    return np.arange(roi_x, roi_x + width, dtype=np.float64)


def to_wavelengths(calibration, roi_x, width):
    if calibration is None:
        return None
    return np.polyval(calibration["coeffs"], pixel_axis(roi_x, width))


def find_spectral_peaks(x_axis, intensity, max_peaks=8):
    y = np.nan_to_num(np.asarray(intensity, dtype=float))
    smooth = np.convolve(y, np.ones(5) / 5, mode="same")
    span = smooth.max() - smooth.min()
    if span <= 0:
        return []
    idx, props = find_peaks(smooth, prominence=0.10 * span, distance=5)
    order = np.argsort(props["prominences"])[::-1][:max_peaks]
    return [
        {"x": round(float(x_axis[idx[i]]), 2), "intensity": round(float(smooth[idx[i]]), 2)}
        for i in sorted(order, key=lambda k: idx[k])
    ]


# ---- dark / reference / absorbance -------------------------------------------------
def compute_absorbance(sample, ref, dark=None):
    """A = -log10((S-D)/(R-D)); NaN where the reference is too dark to be meaningful."""
    s = np.asarray(sample, dtype=float)
    r = np.asarray(ref, dtype=float)
    d = np.zeros_like(s) if dark is None else np.asarray(dark, dtype=float)
    denom = r - d
    valid = denom >= MIN_REF_COUNTS
    ratio = np.clip(s - d, 1e-3, None) / np.clip(denom, 1e-3, None)
    return np.where(valid, -np.log10(ratio), np.nan)


def roi_compatible(item, roi):
    """Reference/dark must cover the same x-range (columns) as the sample ROI."""
    if item is None:
        return False
    return item["roi"][0] == roi[0] and item["roi"][2] == roi[2]


def load_active_reference(db, kind):
    if db is None:
        return None
    d = db.references.find_one({"kind": kind, "active": True}, sort=[("created_at", DESCENDING)])
    if not d:
        return None
    r = d["roi"]
    return {"id": d["_id"], "name": d.get("name", ""), "roi": (r["x"], r["y"], r["w"], r["h"]),
            "intensity": np.array(d["intensity"], dtype=float)}


# ---- preprocessing for ML ------------------------------------------------------------
def baseline_poly(y, order=BASELINE_ORDER, iters=30):
    """Iterative lower-envelope polynomial baseline (modified polyfit)."""
    x = np.linspace(-1, 1, len(y))
    work = y.copy()
    for _ in range(iters):
        fit = np.polyval(np.polyfit(x, work, order), x)
        work = np.minimum(work, fit)
    return np.polyval(np.polyfit(x, work, order), x)


def preprocess(wavelengths, values, signal="absorbance", grid=GRID):
    """Resample onto the common grid, smooth, de-baseline and normalise.

    Returns {"shape": unit-bounded vector, "amplitude": float, "coverage": float} or None.
    absorbance: baseline removed; shape = curve / max(norm, AMP_FLOOR) so that pure noise
                (clean water) stays a small, consistent vector instead of being blown up.
    intensity : shape = curve / its L2 norm (brightness independent), amplitude = 0.
    """
    wl = np.asarray(wavelengths, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(wl) & np.isfinite(y)
    if ok.sum() < 10:
        return None
    wl, y = wl[ok], y[ok]
    wl, idx = np.unique(wl, return_index=True)
    y = y[idx]
    if len(wl) < 10:
        return None
    coverage = float(np.mean((grid >= wl[0]) & (grid <= wl[-1])))
    if coverage < MIN_COVERAGE:
        return None
    yi = savgol_filter(np.interp(grid, wl, y), SMOOTH_WINDOW, 3)
    if signal == "absorbance":
        yi = yi - baseline_poly(yi)
        amp = float(np.linalg.norm(yi))
        shape = yi / max(amp, AMP_FLOOR)
    else:
        norm = float(np.linalg.norm(yi))
        if norm <= 0:
            return None
        shape, amp = yi / norm, 0.0
    return {"shape": shape, "amplitude": amp, "coverage": coverage}


def feature_vector(pre):
    return np.append(pre["shape"], np.log1p(pre["amplitude"]))


def doc_to_arrays(d, signal):
    wl = d.get("wavelengths")
    if not wl:
        return None, "uncalibrated"
    y = d.get(signal)
    if y is None:
        return None, f"no {signal} stored"
    return (np.array(wl, dtype=float), np.array(y, dtype=float)), None


def build_dataset(docs, signal, grid=GRID):
    rows, skipped = [], Counter()
    for d in docs:
        arr, why = doc_to_arrays(d, signal)
        if arr is None:
            skipped[why] += 1
            continue
        pre = preprocess(arr[0], arr[1], signal, grid)
        if pre is None:
            skipped["too short / outside 400-700 nm"] += 1
            continue
        rows.append((d, pre))
    return rows, skipped


# ==================================================================================
# Machine learning
# ==================================================================================
def require_ml():
    if not ML_OK:
        sys.exit("scikit-learn/joblib not installed. Run: pip install scikit-learn joblib")


def make_estimator(kind, seed=42):
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                                   probability=True, class_weight="balanced",
                                                   random_state=seed))
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)


def cross_validate(est, X, y, max_splits=5):
    counts = Counter(y)
    min_count = min(counts.values())
    if min_count < 2:
        return None
    folds = min(max_splits, min_count)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    pred = cross_val_predict(est, X, y, cv=cv)
    classes = sorted(counts)
    return {"accuracy": float(np.mean(pred == y)), "folds": folds, "classes": classes,
            "confusion": confusion_matrix(y, pred, labels=classes).tolist()}


def novelty_threshold(shapes, sigma=3.0):
    """Distance beyond which a spectrum is further from every training spectrum than
    training spectra normally are from each other (leave-one-out nearest neighbour)."""
    if len(shapes) < 3:
        return None
    d = np.linalg.norm(shapes[:, None, :] - shapes[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return float(max(nn.mean() + sigma * nn.std(), 1.5 * np.percentile(nn, 95), NOVELTY_FLOOR))


def train_from_docs(docs, signal, kind, sigma):
    labeled = [d for d in docs if d.get("label")]
    rows, skipped = build_dataset(labeled, signal)
    labels = np.array([d["label"] for d, _ in rows])
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError(f"Need at least 2 labelled classes with usable {signal} spectra "
                         f"(found {dict(counts) or 'none'}). Skipped: {dict(skipped)}")
    X = np.array([feature_vector(p) for _, p in rows])
    shapes = np.array([p["shape"] for _, p in rows])
    est = make_estimator(kind)
    cv = cross_validate(est, X, labels)
    est.fit(X, labels)
    bundle = {
        "version": 1, "estimator": est, "model_type": kind, "signal": signal,
        "classes": [str(c) for c in est.classes_], "grid": GRID,
        "train_shapes": shapes, "novelty_threshold": novelty_threshold(shapes, sigma),
        "train_meta": [{"id": str(d["_id"]) if "_id" in d else "", "sample": d.get("sample", ""),
                        "label": d["label"]} for d, _ in rows],
        "n_samples": len(rows), "class_counts": {str(k): int(v) for k, v in counts.items()},
        "cv": cv, "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return bundle, cv, counts, skipped


def predict_pre(bundle, pre, top_k=3):
    proba = bundle["estimator"].predict_proba(feature_vector(pre).reshape(1, -1))[0]
    order = np.argsort(proba)[::-1][:top_k]
    ranking = [(bundle["classes"][i], float(proba[i])) for i in order]
    dists = np.linalg.norm(bundle["train_shapes"] - pre["shape"], axis=1)
    j = int(np.argmin(dists))
    thr = bundle["novelty_threshold"]
    if thr is not None and dists[j] > thr:
        verdict = "UNKNOWN"
    elif ranking[0][1] < MIN_CONF:
        verdict = "UNCERTAIN"
    else:
        verdict = "KNOWN"
    return {"verdict": verdict, "ranking": ranking, "nn_dist": float(dists[j]),
            "threshold": thr, "nearest": bundle["train_meta"][j], "coverage": pre["coverage"]}


def predict_arrays(bundle, wavelengths, values):
    pre = preprocess(wavelengths, values, bundle["signal"], bundle["grid"])
    return None if pre is None else predict_pre(bundle, pre)


def print_prediction(res):
    if res is None:
        print("Prediction failed: spectrum too short or outside the 400-700 nm window.")
        return
    top, conf = res["ranking"][0]
    print("\n=== Analysis ===")
    if res["verdict"] == "UNKNOWN":
        print(f"Verdict : UNKNOWN - does not resemble anything in the training data "
              f"(closest class guess: {top}, {conf:.0%})")
    elif res["verdict"] == "UNCERTAIN":
        print(f"Verdict : UNCERTAIN - best guess {top} ({conf:.0%}), below {MIN_CONF:.0%} confidence")
    else:
        print(f"Verdict : {top} ({conf:.0%})")
    print("Classes : " + ", ".join(f"{c} {p:.0%}" for c, p in res["ranking"]))
    n = res["nearest"]
    thr = f"{res['threshold']:.3f}" if res["threshold"] is not None else "n/a"
    print(f"Nearest : {n['sample'] or '?'} [{n['label']}] id ...{n['id'][-6:]}  "
          f"distance {res['nn_dist']:.3f} (novelty threshold {thr})")
    if res["coverage"] < 0.8:
        print(f"[note] spectrum covers only {res['coverage']:.0%} of 400-700 nm; accuracy may suffer.")
    print("Screening result only - not a certified water test.\n")


def print_cv(cv, counts):
    print("Samples per class: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if cv is None:
        print("Cross-validation skipped: every class needs at least 2 samples.")
        return
    print(f"Cross-validated accuracy ({cv['folds']}-fold): {cv['accuracy']:.1%}")
    classes = cv["classes"]
    w = max(8, max(len(c) for c in classes) + 1)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(" " * w + "".join(f"{c[:w - 1]:>{w}}" for c in classes))
    for c, row in zip(classes, cv["confusion"]):
        print(f"{c[:w - 1]:<{w}}" + "".join(f"{v:>{w}}" for v in row))
    if min(counts.values()) < 5:
        print("[warn] some classes have <5 samples; treat accuracy as optimistic/unstable.")


def load_model(path=None):
    if not ML_OK:
        return None
    p = Path(path) if path else MODEL_DIR / "latest.joblib"
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        print(f"[warn] could not load model {p}: {e}")
        return None


# ==================================================================================
# Plotting
# ==================================================================================
def plot_capture(cropped, r, g, b, i, x_axis, xlabel, peaks, title, absorbance=None):
    rows = 3 if absorbance is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(9, 3.5 * rows))
    axes[0].imshow(cv2.cvtColor(cropped.astype(np.uint8), cv2.COLOR_BGR2RGB), aspect="auto")
    axes[0].set_title("Selected ROI")
    ax2 = axes[1]
    ax2.plot(x_axis, r, "r", label="Red")
    ax2.plot(x_axis, g, "g", label="Green")
    ax2.plot(x_axis, b, "b", label="Blue")
    ax2.plot(x_axis, i, "k", label="Intensity")
    for p in peaks:
        ax2.axvline(p["x"], color="gray", ls=":", lw=0.8)
        ax2.annotate(f"{p['x']:.1f}", (p["x"], p["intensity"]),
                     textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Intensity (0-255)")
    ax2.legend(loc="upper left")
    if absorbance is not None:
        axes[2].plot(x_axis, absorbance, color="purple")
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylabel("Absorbance")
        axes[2].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_docs(docs, normalize=True, signal="intensity"):
    plt.figure(figsize=(9, 5))
    calibrated = all(d.get("wavelengths") for d in docs)
    plotted = 0
    for d in docs:
        if d.get(signal) is None:
            print(f"Skipping {d['_id']}: no {signal} stored.")
            continue
        y = np.array(d[signal], dtype=float)
        if normalize and np.isfinite(y).any() and np.nanmax(np.abs(y)) > 0:
            y = y / np.nanmax(np.abs(y))
        x = np.array(d["wavelengths"]) if calibrated else np.array(d["pixel_x"])
        plt.plot(x, y, label=f"{d['sample']} [{d.get('label') or '-'}] ({str(d['_id'])[-6:]})")
        plotted += 1
    if not plotted:
        plt.close()
        return
    plt.xlabel("Wavelength (nm)" if calibrated else "Pixel (uncalibrated)")
    plt.ylabel(("Normalized " if normalize else "") + signal)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================================================================================
# Live capture workflow
# ==================================================================================
class Session:
    def __init__(self):
        self.roi = None
        self.calibration = None
        self.dark = None
        self.ref = None
        self.model = None
        self.cam_settings = {}


def ask_float(prompt):
    s = input(prompt).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        print("  (not a number - skipped)")
        return None


def prompt_metadata():
    return {
        "sample": input("  Sample name: ").strip() or "unnamed",
        "label": input("  Class label for ML (e.g. clean, salty, copper; blank = none): ")
                 .strip().lower() or None,
        "liquid": input("  Liquid type (e.g. tap water): ").strip() or None,
        "impurity": input("  Known impurity (e.g. CuSO4, NaCl): ").strip() or None,
        "concentration_ppm": ask_float("  Concentration ppm (blank = unknown): "),
        "temperature_c": ask_float("  Temperature C: "),
        "ph": ask_float("  pH: "),
        "tds_ppm": ask_float("  TDS meter reading ppm: "),
        "location": input("  Location: ").strip() or None,
        "tags": [t.strip() for t in input("  Tags (comma separated): ").split(",") if t.strip()],
        "notes": input("  Notes: ").strip(),
    }


def do_calibration(cap, roi, n_frames, db):
    cropped = capture_averaged(cap, roi, n_frames)
    if cropped is None:
        print("Could not read camera.")
        return None
    x, _, w, _ = roi
    _, _, _, intensity = extract_spectrum(cropped)
    px = pixel_axis(x, w)

    fig = plt.figure(figsize=(9, 5))
    plt.plot(px, intensity, "k")
    plt.title("Click on known spectral lines (left-to-right), then press ENTER")
    plt.xlabel("Pixel")
    clicks = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    if len(clicks) < 2:
        print("Need at least 2 points. Calibration cancelled.")
        return None

    pixels, wavelengths = [], []
    for k, (cx, _) in enumerate(clicks, 1):
        try:
            wl = float(input(f"  Wavelength (nm) for point {k} at pixel {cx:.1f}: "))
        except ValueError:
            print("Invalid number. Calibration cancelled.")
            return None
        pixels.append(cx)
        wavelengths.append(wl)

    coeffs, degree, rms = fit_calibration(pixels, wavelengths)
    print(f"Calibration fit: degree={degree}, RMS error={rms:.2f} nm")
    cal = {
        "name": input("  Calibration name (e.g. 'CFL bulb'): ").strip() or "unnamed",
        "created_at": datetime.now(timezone.utc),
        "pixels": pixels, "wavelengths": wavelengths, "coeffs": coeffs,
        "degree": degree, "rms_error_nm": rms, "active": True,
    }
    if db is not None:
        db.calibrations.update_many({"active": True}, {"$set": {"active": False}})
        cal["_id"] = db.calibrations.insert_one(cal).inserted_id
        print(f"Calibration saved (id {cal['_id']}).")
    return cal


def do_reference(cap, roi, n_frames, kind, session, db):
    """Capture a 'dark' (no light) or 'reference' (blank / distilled water) spectrum."""
    if kind == "dark":
        input("Block ALL light into the spectrometer, then press ENTER... ")
    else:
        input("Put distilled water / blank in the holder with the light ON, then press ENTER... ")
    cropped = capture_averaged(cap, roi, n_frames)
    if cropped is None:
        print("Could not read camera.")
        return None
    r, g, b, i = extract_spectrum(cropped)
    if kind == "reference" and max(r.max(), g.max(), b.max()) >= SATURATION:
        print("[warn] Reference is saturated - lower the light/exposure and redo it.")
    x, y, w, h = roi
    name = input(f"  {kind.capitalize()} name (e.g. 'distilled water, cuvette A'): ").strip() or kind
    item = {"id": None, "name": name, "roi": (x, y, w, h), "intensity": i.astype(float)}
    if db is not None:
        db.references.update_many({"kind": kind, "active": True}, {"$set": {"active": False}})
        item["id"] = db.references.insert_one({
            "kind": kind, "name": name, "created_at": datetime.now(timezone.utc),
            "roi": {"x": x, "y": y, "w": w, "h": h}, "intensity": i.tolist(),
            "camera_settings": session.cam_settings, "active": True}).inserted_id
        print(f"{kind.capitalize()} saved (id {item['id']}).")
    else:
        print(f"{kind.capitalize()} stored for this session only (no database).")
    return item


def do_capture(cap, roi, n_frames, camera_index, session, db):
    cropped = capture_averaged(cap, roi, n_frames)
    if cropped is None:
        print("Could not read camera.")
        return
    x, y, w, h = roi
    calibration = session.calibration
    r, g, b, i = extract_spectrum(cropped)
    saturated = bool(max(r.max(), g.max(), b.max()) >= SATURATION)
    if saturated:
        print("[warn] Sensor saturation detected - reduce exposure/light for accurate data.")

    wl = to_wavelengths(calibration, x, w)
    x_axis = wl if wl is not None else pixel_axis(x, w)
    xlabel = "Wavelength (nm)" if wl is not None else "Pixel (uncalibrated)"
    unit = "nm" if wl is not None else "px"
    peaks = find_spectral_peaks(x_axis, i)

    # absorbance (needs a reference with the same ROI columns)
    absorbance, ref_id, dark_id = None, None, None
    if session.ref is not None:
        if roi_compatible(session.ref, roi):
            dark = session.dark if roi_compatible(session.dark, roi) else None
            if session.dark is not None and dark is None:
                print("[warn] dark frame ROI differs from current ROI - ignoring dark.")
            absorbance = compute_absorbance(i, session.ref["intensity"],
                                            dark["intensity"] if dark else None)
            ref_id = session.ref["id"]
            dark_id = dark["id"] if dark else None
        else:
            print("[warn] reference ROI differs from current ROI - press 'f' to recapture it.")
    else:
        print("[note] no reference captured (press 'f'): absorbance unavailable, "
              "ML works best with absorbance.")

    legacy = rgb_to_wavelength(r.mean(), g.mean(), b.mean())
    if peaks:
        print("Peaks: " + ", ".join(f"{p['x']:.1f}{unit}" for p in peaks))
    title = (f"Dominant peak: {max(peaks, key=lambda p: p['intensity'])['x']:.1f} {unit}"
             if peaks else "Spectrum")
    plot_capture(cropped, r, g, b, i, x_axis, xlabel, peaks, title, absorbance)

    # ML analysis of the unknown sample
    if session.model is not None:
        if wl is None:
            print("[note] calibrate first (press 'c') to use the ML model.")
        else:
            sig = session.model["signal"]
            vals = absorbance if sig == "absorbance" else i
            if vals is None:
                print(f"[note] model needs {sig}; capture a reference ('f') first.")
            else:
                print_prediction(predict_arrays(session.model, wl, vals))

    if db is None:
        return
    if input("Save to MongoDB? [Y/n]: ").strip().lower() == "n":
        return

    doc = prompt_metadata()
    doc.update({
        "created_at": datetime.now(timezone.utc),
        "camera_index": camera_index, "camera_settings": session.cam_settings,
        "frames_averaged": n_frames, "roi": {"x": x, "y": y, "w": w, "h": h},
        "pixel_x": pixel_axis(x, w).tolist(),
        "intensity": i.tolist(), "r": r.tolist(), "g": g.tolist(), "b": b.tolist(),
        "absorbance": absorbance.tolist() if absorbance is not None else None,
        "reference_id": ref_id, "dark_id": dark_id,
        "wavelengths": wl.tolist() if wl is not None else None,
        "calibration_id": calibration["_id"] if calibration and "_id" in calibration else None,
        "peaks": peaks, "peaks_unit": unit,
        "legacy_hue_wavelength_nm": legacy, "saturated": saturated,
    })
    result = db.spectra.insert_one(doc)
    print(f"Saved spectrum with id {result.inserted_id}")


def run_live(args):
    db = try_get_db()
    session = Session()
    session.calibration = load_active_calibration(db)
    session.dark = load_active_reference(db, "dark")
    session.ref = load_active_reference(db, "reference")
    session.model = load_model()
    if session.calibration:
        print(f"Loaded calibration '{session.calibration['name']}' "
              f"(RMS {session.calibration['rms_error_nm']:.2f} nm)")
    else:
        print("No calibration loaded - press 'c' to calibrate.")
    print(f"Reference: {session.ref['name'] if session.ref else 'none (press f)'} | "
          f"Dark: {session.dark['name'] if session.dark else 'none (press d)'}")
    if session.model:
        print(f"ML model loaded: {session.model['model_type']} on {session.model['signal']}, "
              f"classes {session.model['classes']}")
    else:
        print("No ML model loaded (run 'train' after collecting labelled spectra).")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Failed to open camera. Try a different --camera index.")
        return
    if args.lock_camera:
        session.cam_settings = lock_camera(cap, args.exposure, args.gain, args.wb)
        print(f"Camera locked: {session.cam_settings}")
    else:
        session.cam_settings = camera_settings(cap)
        print("[tip] use --lock-camera so auto exposure/white balance don't change your spectra.")

    print("Keys: [r] ROI  [c] calibrate  [d] dark  [f] reference  [s] capture  [q] quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        if session.roi:
            x, y, w, h = session.roi
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.imshow("roi", frame[y:y + h, x:x + w])
        cv2.imshow("frame", display)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("r"):
            sel = cv2.selectROI("frame", frame)
            if sel[2] > 0 and sel[3] > 0:
                session.roi = tuple(int(v) for v in sel)
        elif k in (ord("s"), ord("c"), ord("d"), ord("f")):
            if session.roi is None:
                print("Select an ROI first (press 'r').")
            elif k == ord("c"):
                session.calibration = do_calibration(cap, session.roi, args.avg, db) \
                    or session.calibration
            elif k == ord("d"):
                session.dark = do_reference(cap, session.roi, args.avg, "dark", session, db) \
                    or session.dark
            elif k == ord("f"):
                session.ref = do_reference(cap, session.roi, args.avg, "reference", session, db) \
                    or session.ref
            else:
                do_capture(cap, session.roi, args.avg, args.camera, session, db)

    cap.release()
    cv2.destroyAllWindows()


# ==================================================================================
# Database commands
# ==================================================================================
def cmd_list(args):
    db = get_db()
    query = {}
    if args.sample:
        query["sample"] = args.sample
    if args.label:
        query["label"] = args.label.lower()
    docs = db.spectra.find(query, {"sample": 1, "label": 1, "created_at": 1, "peaks": 1,
                                   "peaks_unit": 1, "wavelengths": 1, "absorbance": 1}
                           ).sort("created_at", DESCENDING).limit(args.limit)
    print(f"{'ID':26} {'Created (UTC)':20} {'Sample':18} {'Label':12} {'Cal':3} {'Abs':3} Top peak")
    for d in docs:
        top = max(d["peaks"], key=lambda p: p["intensity"]) if d.get("peaks") else None
        top_s = f"{top['x']:.1f} {d.get('peaks_unit', '')}" if top else "-"
        print(f"{d['_id']!s:26} {d['created_at']:%Y-%m-%d %H:%M:%S}  "
              f"{d['sample'][:18]:18} {(d.get('label') or '-')[:12]:12} "
              f"{'yes' if d.get('wavelengths') else 'no':3} "
              f"{'yes' if d.get('absorbance') else 'no':3} {top_s}")


def fetch_docs(db, ids):
    docs = []
    for s in ids:
        try:
            d = db.spectra.find_one({"_id": ObjectId(s)})
        except Exception:
            d = None
        if d is None:
            print(f"Not found: {s}")
        else:
            docs.append(d)
    return docs


def cmd_plot(args):
    docs = fetch_docs(get_db(), [args.id])
    if docs:
        plot_docs(docs, normalize=False, signal=args.signal)


def cmd_compare(args):
    docs = fetch_docs(get_db(), args.ids)
    if docs:
        plot_docs(docs, normalize=True, signal=args.signal)


def _col(d, key, n):
    v = d.get(key)
    return v if v is not None else [""] * n


def cmd_export(args):
    docs = fetch_docs(get_db(), [args.id])
    if not docs:
        return
    d = docs[0]
    n = len(d["intensity"])
    out = args.output or f"spectrum_{d['_id']}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pixel", "wavelength_nm", "r", "g", "b", "intensity", "absorbance"])
        for row in zip(d["pixel_x"], _col(d, "wavelengths", n), _col(d, "r", n),
                       _col(d, "g", n), _col(d, "b", n), d["intensity"],
                       _col(d, "absorbance", n)):
            wr.writerow(row)
    print(f"Wrote {out}")


def cmd_delete(args):
    db = get_db()
    d = db.spectra.find_one({"_id": oid(args.id)}, {"sample": 1, "label": 1})
    if not d:
        print("Not found.")
        return
    if not args.yes and input(f"Delete '{d['sample']}' [{d.get('label') or '-'}]? [y/N]: "
                              ).strip().lower() != "y":
        print("Cancelled.")
        return
    db.spectra.delete_one({"_id": d["_id"]})
    print("Deleted.")


def cmd_annotate(args):
    db = get_db()
    upd = {}
    for field in ("sample", "liquid", "impurity", "location", "notes"):
        if getattr(args, field) is not None:
            upd[field] = getattr(args, field)
    if args.label is not None:
        upd["label"] = args.label.strip().lower() or None
    for field, attr in (("concentration_ppm", "concentration"), ("temperature_c", "temperature"),
                        ("ph", "ph"), ("tds_ppm", "tds")):
        if getattr(args, attr) is not None:
            upd[field] = getattr(args, attr)
    if args.tags is not None:
        upd["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if not upd:
        print("Nothing to update.")
        return
    res = db.spectra.update_one({"_id": oid(args.id)}, {"$set": upd})
    print("Updated." if res.matched_count else "Not found.")


# ---- import ---------------------------------------------------------------------------
def read_spectrum_csv(path):
    """Read a CSV with wavelength_nm and intensity and/or absorbance (r,g,b,pixel optional)."""
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        cols = {(n or "").strip().lower(): n for n in (rd.fieldnames or [])}
        wl_key = cols.get("wavelength_nm") or cols.get("wavelength")
        if not wl_key or not (cols.get("intensity") or cols.get("absorbance")):
            raise ValueError("needs a wavelength_nm column and an intensity and/or absorbance column")
        data = {k: [] for k in ("wl", "intensity", "absorbance", "r", "g", "b", "pixel")}
        keys = {"wl": wl_key, "intensity": cols.get("intensity"), "absorbance": cols.get("absorbance"),
                "r": cols.get("r"), "g": cols.get("g"), "b": cols.get("b"), "pixel": cols.get("pixel")}
        for row in rd:
            for k, col in keys.items():
                if col is None:
                    continue
                try:
                    data[k].append(float(row[col]))
                except (TypeError, ValueError):
                    data[k].append(float("nan"))
    n = len(data["wl"])
    if n < 10:
        raise ValueError("fewer than 10 rows")
    return data, n


def cmd_import(args):
    db = get_db()
    root = Path(args.path)
    files = sorted(root.rglob("*.csv")) if root.is_dir() else [root]
    if not files:
        sys.exit("No CSV files found.")
    done = 0
    for p in files:
        try:
            data, n = read_spectrum_csv(p)
        except (ValueError, OSError) as e:
            print(f"Skipping {p}: {e}")
            continue
        label = args.label
        if label is None and root.is_dir() and p.parent != root:
            label = p.parent.name  # dataset/<label>/<file>.csv
        label = label.strip().lower() if label else None
        nan = [float("nan")] * n
        intensity = data["intensity"] or nan
        peak_src = data["intensity"] or data["absorbance"]
        doc = {
            "sample": args.sample or p.stem, "label": label, "liquid": args.liquid,
            "impurity": args.impurity, "concentration_ppm": None, "temperature_c": None,
            "ph": None, "tds_ppm": None, "location": None, "tags": ["imported"],
            "notes": f"imported from {p.name}", "created_at": datetime.now(timezone.utc),
            "camera_index": None, "camera_settings": {}, "frames_averaged": None, "roi": None,
            "pixel_x": data["pixel"] or list(range(n)), "intensity": intensity,
            "r": data["r"] or None, "g": data["g"] or None, "b": data["b"] or None,
            "absorbance": data["absorbance"] or None, "reference_id": None, "dark_id": None,
            "wavelengths": data["wl"], "calibration_id": None,
            "peaks": find_spectral_peaks(np.array(data["wl"]), np.array(peak_src)),
            "peaks_unit": "nm", "legacy_hue_wavelength_nm": None, "saturated": False,
            "imported": True,
        }
        db.spectra.insert_one(doc)
        done += 1
    print(f"Imported {done}/{len(files)} file(s).")


# ---- dataset export ---------------------------------------------------------------------
def cmd_export_dataset(args):
    db = get_db()
    query = {} if args.all else {"label": {"$nin": [None, ""]}}
    rows, skipped = build_dataset(list(db.spectra.find(query)), args.signal)
    if not rows:
        sys.exit(f"No usable spectra. Skipped: {dict(skipped)}")
    out = Path(args.output)
    ids = [str(d["_id"]) for d, _ in rows]
    labels = [d.get("label") or "" for d, _ in rows]
    shapes = np.array([p["shape"] for _, p in rows])
    amps = np.array([p["amplitude"] for _, p in rows])
    if out.suffix.lower() == ".npz":
        np.savez(out, X=np.array([feature_vector(p) for _, p in rows]), shapes=shapes,
                 amplitude=amps, y=np.array(labels), ids=np.array(ids), grid=GRID,
                 signal=args.signal)
    else:
        with open(out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["id", "label", "sample", "liquid", "impurity", "concentration_ppm",
                         "amplitude"] + [f"nm_{int(g)}" for g in GRID])
            for (d, p), i in zip(rows, ids):
                wr.writerow([i, d.get("label") or "", d.get("sample", ""), d.get("liquid") or "",
                             d.get("impurity") or "", d.get("concentration_ppm") or "",
                             round(p["amplitude"], 5)] + [round(float(v), 6) for v in p["shape"]])
    print(f"Wrote {out}: {len(rows)} spectra x {len(GRID)} wavelengths ({args.signal}). "
          f"Skipped: {dict(skipped) or 'none'}")


# ---- ML commands -------------------------------------------------------------------------
def cmd_evaluate(args):
    require_ml()
    db = get_db()
    try:
        _, cv, counts, skipped = train_from_docs(list(db.spectra.find()), args.signal,
                                                 args.model, args.novelty_sigma)
    except ValueError as e:
        sys.exit(str(e))
    print(f"Model: {args.model} | signal: {args.signal} | skipped: {dict(skipped) or 'none'}")
    print_cv(cv, counts)


def cmd_train(args):
    require_ml()
    db = get_db()
    try:
        bundle, cv, counts, skipped = train_from_docs(list(db.spectra.find()), args.signal,
                                                      args.model, args.novelty_sigma)
    except ValueError as e:
        sys.exit(str(e))
    print(f"Trained {args.model} on {bundle['n_samples']} spectra ({args.signal}). "
          f"Skipped: {dict(skipped) or 'none'}")
    print_cv(cv, counts)
    thr = bundle["novelty_threshold"]
    print(f"Novelty threshold: {thr:.3f}" if thr is not None
          else "Novelty detection disabled (need >=3 spectra).")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MODEL_DIR / f"model_{stamp}.joblib"
    joblib.dump(bundle, path)
    joblib.dump(bundle, MODEL_DIR / "latest.joblib")
    meta = {k: bundle[k] for k in ("model_type", "signal", "classes", "n_samples",
                                   "class_counts", "novelty_threshold", "cv", "created_at")}
    meta["file"] = str(path)
    db.models.insert_one(meta)
    print(f"Saved {path} (and models/latest.joblib)")


def cmd_predict(args):
    require_ml()
    bundle = load_model(args.model)
    if bundle is None:
        sys.exit("No model found. Run: python spectrometer.py train")
    docs = fetch_docs(get_db(), [args.id])
    if not docs:
        return
    arr, why = doc_to_arrays(docs[0], bundle["signal"])
    if arr is None:
        sys.exit(f"Cannot predict: {why} (model uses {bundle['signal']}).")
    print_prediction(predict_arrays(bundle, arr[0], arr[1]))


def cmd_match(args):
    db = get_db()
    docs = fetch_docs(db, [args.id])
    if not docs:
        return
    query_doc = docs[0]
    arr, why = doc_to_arrays(query_doc, args.signal)
    if arr is None:
        sys.exit(f"Cannot match: {why}.")
    q = preprocess(arr[0], arr[1], args.signal)
    if q is None:
        sys.exit("Query spectrum too short / outside 400-700 nm.")
    others = [d for d in db.spectra.find({"_id": {"$ne": query_doc["_id"]}})]
    rows, _ = build_dataset(others, args.signal)
    if not rows:
        sys.exit("No other usable spectra to compare against.")
    scored = sorted(((float(np.linalg.norm(p["shape"] - q["shape"])), d) for d, p in rows),
                    key=lambda t: t[0])[:args.top]
    print(f"Closest to '{query_doc['sample']}' [{query_doc.get('label') or '-'}] ({args.signal}):")
    for dist, d in scored:
        sim = max(0.0, 1.0 - dist / 2.0)
        print(f"  {sim:6.1%}  {d['sample'][:22]:22} [{(d.get('label') or '-')[:12]:12}] "
              f"id ...{str(d['_id'])[-6:]}  (distance {dist:.3f})")


# ==================================================================================
def main():
    p = argparse.ArgumentParser(description="AquaSpectra: DIY spectrometer with ML + MongoDB")
    sub = p.add_subparsers(dest="cmd")

    def add_signal(sp, default="absorbance"):
        sp.add_argument("--signal", choices=["absorbance", "intensity"], default=default)

    live = sub.add_parser("live", help="live camera capture (default)")
    live.add_argument("--camera", type=int, default=1, help="camera index (0 or 1)")
    live.add_argument("--avg", type=int, default=10, help="frames to average per capture")
    live.add_argument("--lock-camera", action="store_true",
                      help="disable auto exposure / white balance")
    live.add_argument("--exposure", type=float, help="manual exposure (driver specific units)")
    live.add_argument("--gain", type=float, help="manual gain")
    live.add_argument("--wb", type=float, help="manual white-balance temperature (K)")
    live.set_defaults(func=run_live)

    ls = sub.add_parser("list", help="list saved spectra")
    ls.add_argument("--sample")
    ls.add_argument("--label")
    ls.add_argument("--limit", type=int, default=20)
    ls.set_defaults(func=cmd_list)

    pl = sub.add_parser("plot", help="plot one saved spectrum")
    pl.add_argument("id")
    add_signal(pl, "intensity")
    pl.set_defaults(func=cmd_plot)

    cp = sub.add_parser("compare", help="overlay several saved spectra")
    cp.add_argument("ids", nargs="+")
    add_signal(cp, "intensity")
    cp.set_defaults(func=cmd_compare)

    ex = sub.add_parser("export", help="export a spectrum to CSV")
    ex.add_argument("id")
    ex.add_argument("-o", "--output")
    ex.set_defaults(func=cmd_export)

    dl = sub.add_parser("delete", help="delete a saved spectrum")
    dl.add_argument("id")
    dl.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    dl.set_defaults(func=cmd_delete)

    an = sub.add_parser("annotate", help="edit labels/metadata of a saved spectrum")
    an.add_argument("id")
    for name in ("label", "sample", "liquid", "impurity", "location", "notes", "tags"):
        an.add_argument(f"--{name}")
    for name in ("concentration", "temperature", "ph", "tds"):
        an.add_argument(f"--{name}", type=float)
    an.set_defaults(func=cmd_annotate)

    im = sub.add_parser("import", help="import spectra from a CSV file or a folder of CSVs "
                                       "(folder/<label>/file.csv uses the folder as the label)")
    im.add_argument("path")
    for name in ("label", "sample", "liquid", "impurity"):
        im.add_argument(f"--{name}")
    im.set_defaults(func=cmd_import)

    ed = sub.add_parser("export-dataset", help="export preprocessed dataset (.csv or .npz)")
    ed.add_argument("-o", "--output", required=True)
    ed.add_argument("--all", action="store_true", help="include unlabelled spectra")
    add_signal(ed)
    ed.set_defaults(func=cmd_export_dataset)

    for name, fn, hlp in (("train", cmd_train, "train and save a model"),
                          ("evaluate", cmd_evaluate, "cross-validate without saving")):
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("--model", choices=["rf", "svm"], default="rf")
        sp.add_argument("--novelty-sigma", type=float, default=3.0,
                        help="higher = fewer spectra flagged UNKNOWN")
        add_signal(sp)
        sp.set_defaults(func=fn)

    pr = sub.add_parser("predict", help="analyse a stored spectrum with the trained model")
    pr.add_argument("id")
    pr.add_argument("--model", help="path to a .joblib (default models/latest.joblib)")
    pr.set_defaults(func=cmd_predict)

    mt = sub.add_parser("match", help="find the most similar stored spectra")
    mt.add_argument("id")
    mt.add_argument("--top", type=int, default=5)
    add_signal(mt)
    mt.set_defaults(func=cmd_match)

    args = p.parse_args()
    if args.cmd is None:
        args = p.parse_args(["live"])
    try:
        args.func(args)
    except PyMongoError as e:
        sys.exit(f"MongoDB error: {e}")


if __name__ == "__main__":
    main()
