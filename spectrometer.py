#!/usr/bin/env python3
"""
DIY Spectrometer v2 - camera capture + wavelength calibration + MongoDB storage.

Live-view keys:
    r  select ROI (drag a box over the spectrum band, press ENTER)
    c  calibrate wavelength axis (click known spectral lines)
    s  capture spectrum, preview, and (optionally) save to MongoDB
    q  quit

CLI commands:
    python spectrometer.py live [--camera 0] [--avg 10]
    python spectrometer.py list [--sample NAME]
    python spectrometer.py plot ID
    python spectrometer.py compare ID1 ID2 ...
    python spectrometer.py export ID [-o file.csv]
    python spectrometer.py delete ID

Config (environment variables):
    MONGO_URI   default mongodb://localhost:27017
    MONGO_DB    default spectrometer
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timezone

import cv2
import matplotlib.pyplot as plt
import numpy as np
from bson import ObjectId
from pymongo import DESCENDING, MongoClient
from pymongo.errors import PyMongoError
from scipy.signal import find_peaks

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "spectrometer")


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")  # fail fast if the server is unreachable
    db = client[MONGO_DB]
    db.spectra.create_index([("created_at", DESCENDING)])
    db.spectra.create_index("sample")
    db.calibrations.create_index([("active", 1), ("created_at", DESCENDING)])
    return db


def try_get_db():
    try:
        return get_db()
    except PyMongoError as e:
        print(f"[warn] MongoDB unavailable ({e.__class__.__name__}). "
              f"Running without storage.")
        return None


def load_active_calibration(db):
    if db is None:
        return None
    return db.calibrations.find_one({"active": True}, sort=[("created_at", DESCENDING)])


# --------------------------------------------------------------------------
# Spectrum processing
# --------------------------------------------------------------------------
def rgb_to_wavelength(r, g, b):
    """Legacy hue-based estimate from the first project (kept for reference only)."""
    hsv = cv2.cvtColor(np.uint8([[[int(r), int(g), int(b)]]]), cv2.COLOR_RGB2HSV)[0][0]
    return round(750 - (hsv[0] / 179.0) * (750 - 380), 1)


def capture_averaged(cap, roi, n_frames):
    """Average n frames of the ROI to reduce sensor noise."""
    x, y, w, h = roi
    acc = None
    count = 0
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
    intensity = (r + g + b) / 3.0
    return r, g, b, intensity


def fit_calibration(pixels, wavelengths):
    degree = 1 if len(pixels) == 2 else 2
    coeffs = np.polyfit(pixels, wavelengths, degree)
    residual = np.polyval(coeffs, pixels) - np.array(wavelengths)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return coeffs.tolist(), degree, rms


def pixel_axis(roi_x, width):
    return np.arange(roi_x, roi_x + width, dtype=np.float64)


def to_wavelengths(calibration, roi_x, width):
    if calibration is None:
        return None
    return np.polyval(calibration["coeffs"], pixel_axis(roi_x, width))


def find_spectral_peaks(x_axis, intensity, max_peaks=8):
    smooth = np.convolve(intensity, np.ones(5) / 5, mode="same")
    span = smooth.max() - smooth.min()
    if span <= 0:
        return []
    idx, props = find_peaks(smooth, prominence=0.10 * span, distance=5)
    order = np.argsort(props["prominences"])[::-1][:max_peaks]
    return [
        {"x": round(float(x_axis[idx[i]]), 2), "intensity": round(float(smooth[idx[i]]), 2)}
        for i in sorted(order, key=lambda k: idx[k])
    ]


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_capture(cropped, r, g, b, i, x_axis, xlabel, peaks, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    ax1.imshow(cv2.cvtColor(cropped.astype(np.uint8), cv2.COLOR_BGR2RGB), aspect="auto")
    ax1.set_title("Selected ROI")
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
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_docs(docs, normalize=True):
    plt.figure(figsize=(9, 5))
    calibrated = all(d.get("wavelengths") for d in docs)
    for d in docs:
        y = np.array(d["intensity"])
        if normalize and y.max() > 0:
            y = y / y.max()
        x = np.array(d["wavelengths"]) if calibrated else np.array(d["pixel_x"])
        plt.plot(x, y, label=f"{d['sample']} ({str(d['_id'])[-6:]})")
    plt.xlabel("Wavelength (nm)" if calibrated else "Pixel (uncalibrated)")
    plt.ylabel("Normalized intensity" if normalize else "Intensity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# Live capture workflow
# --------------------------------------------------------------------------
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
        "pixels": pixels,
        "wavelengths": wavelengths,
        "coeffs": coeffs,
        "degree": degree,
        "rms_error_nm": rms,
        "active": True,
    }
    if db is not None:
        db.calibrations.update_many({"active": True}, {"$set": {"active": False}})
        cal["_id"] = db.calibrations.insert_one(cal).inserted_id
        print(f"Calibration saved (id {cal['_id']}).")
    return cal


def do_capture(cap, roi, n_frames, camera_index, calibration, db):
    cropped = capture_averaged(cap, roi, n_frames)
    if cropped is None:
        print("Could not read camera.")
        return
    x, y, w, h = roi
    r, g, b, i = extract_spectrum(cropped)

    if max(r.max(), g.max(), b.max()) >= 250:
        print("[warn] Sensor saturation detected - reduce exposure/light for accurate data.")

    wl = to_wavelengths(calibration, x, w)
    x_axis = wl if wl is not None else pixel_axis(x, w)
    xlabel = "Wavelength (nm)" if wl is not None else "Pixel (uncalibrated)"
    peaks = find_spectral_peaks(x_axis, i)
    legacy = rgb_to_wavelength(cropped[:, :, 2].mean(), cropped[:, :, 1].mean(),
                               cropped[:, :, 0].mean())

    if peaks:
        unit = "nm" if wl is not None else "px"
        print("Peaks: " + ", ".join(f"{p['x']:.1f}{unit}" for p in peaks))
    title = (f"Dominant peak: {max(peaks, key=lambda p: p['intensity'])['x']:.1f} "
             f"{'nm' if wl is not None else 'px'}" if peaks else "Spectrum")
    plot_capture(cropped, r, g, b, i, x_axis, xlabel, peaks, title)

    if db is None:
        return
    if input("Save to MongoDB? [Y/n]: ").strip().lower() == "n":
        return

    doc = {
        "sample": input("  Sample name: ").strip() or "unnamed",
        "notes": input("  Notes (optional): ").strip(),
        "created_at": datetime.now(timezone.utc),
        "camera_index": camera_index,
        "frames_averaged": n_frames,
        "roi": {"x": x, "y": y, "w": w, "h": h},
        "pixel_x": pixel_axis(x, w).tolist(),
        "intensity": i.tolist(),
        "r": r.tolist(),
        "g": g.tolist(),
        "b": b.tolist(),
        "wavelengths": wl.tolist() if wl is not None else None,
        "calibration_id": calibration["_id"] if calibration and "_id" in calibration else None,
        "peaks": peaks,
        "peaks_unit": "nm" if wl is not None else "px",
        "legacy_hue_wavelength_nm": legacy,
        "saturated": bool(max(r.max(), g.max(), b.max()) >= 250),
    }
    result = db.spectra.insert_one(doc)
    print(f"Saved spectrum with id {result.inserted_id}")


def run_live(args):
    db = try_get_db()
    calibration = load_active_calibration(db)
    if calibration:
        print(f"Loaded calibration '{calibration['name']}' "
              f"(RMS {calibration['rms_error_nm']:.2f} nm)")
    else:
        print("No calibration loaded - press 'c' to calibrate.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Failed to open camera. Try a different --camera index.")
        return

    roi = None
    print("Keys: [r] ROI  [c] calibrate  [s] capture  [q] quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        if roi:
            x, y, w, h = roi
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.imshow("roi", frame[y:y + h, x:x + w])
        cv2.imshow("frame", display)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("r"):
            sel = cv2.selectROI("frame", frame)
            if sel[2] > 0 and sel[3] > 0:
                roi = tuple(int(v) for v in sel)
        elif k in (ord("s"), ord("c")):
            if roi is None:
                print("Select an ROI first (press 'r').")
            elif k == ord("c"):
                new_cal = do_calibration(cap, roi, args.avg, db)
                calibration = new_cal or calibration
            else:
                do_capture(cap, roi, args.avg, args.camera, calibration, db)

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# Database commands
# --------------------------------------------------------------------------
def cmd_list(args):
    db = get_db()
    query = {"sample": args.sample} if args.sample else {}
    docs = db.spectra.find(query, {"sample": 1, "created_at": 1, "peaks": 1,
                                   "peaks_unit": 1, "wavelengths": 1}
                           ).sort("created_at", DESCENDING).limit(args.limit)
    print(f"{'ID':26} {'Created (UTC)':20} {'Sample':20} {'Cal':4} Top peak")
    for d in docs:
        top = max(d["peaks"], key=lambda p: p["intensity"]) if d.get("peaks") else None
        top_s = f"{top['x']:.1f} {d.get('peaks_unit', '')}" if top else "-"
        print(f"{d['_id']!s:26} {d['created_at']:%Y-%m-%d %H:%M:%S}  "
              f"{d['sample'][:20]:20} {'yes' if d.get('wavelengths') else 'no':4} {top_s}")


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
        plot_docs(docs, normalize=False)


def cmd_compare(args):
    docs = fetch_docs(get_db(), args.ids)
    if docs:
        plot_docs(docs, normalize=True)


def cmd_export(args):
    docs = fetch_docs(get_db(), [args.id])
    if not docs:
        return
    d = docs[0]
    out = args.output or f"spectrum_{d['_id']}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pixel", "wavelength_nm", "r", "g", "b", "intensity"])
        wls = d.get("wavelengths") or [""] * len(d["intensity"])
        for row in zip(d["pixel_x"], wls, d["r"], d["g"], d["b"], d["intensity"]):
            wr.writerow(row)
    print(f"Wrote {out}")


def cmd_delete(args):
    db = get_db()
    res = db.spectra.delete_one({"_id": ObjectId(args.id)})
    print("Deleted." if res.deleted_count else "Not found.")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="DIY spectrometer with MongoDB storage")
    sub = p.add_subparsers(dest="cmd")

    live = sub.add_parser("live", help="live camera capture (default)")
    live.add_argument("--camera", type=int, default=1, help="camera index (0 or 1)")
    live.add_argument("--avg", type=int, default=10, help="frames to average per capture")
    live.set_defaults(func=run_live)

    ls = sub.add_parser("list", help="list saved spectra")
    ls.add_argument("--sample")
    ls.add_argument("--limit", type=int, default=20)
    ls.set_defaults(func=cmd_list)

    pl = sub.add_parser("plot", help="plot one saved spectrum")
    pl.add_argument("id")
    pl.set_defaults(func=cmd_plot)

    cp = sub.add_parser("compare", help="overlay several saved spectra")
    cp.add_argument("ids", nargs="+")
    cp.set_defaults(func=cmd_compare)

    ex = sub.add_parser("export", help="export a spectrum to CSV")
    ex.add_argument("id")
    ex.add_argument("-o", "--output")
    ex.set_defaults(func=cmd_export)

    dl = sub.add_parser("delete", help="delete a saved spectrum")
    dl.add_argument("id")
    dl.set_defaults(func=cmd_delete)

    args = p.parse_args()
    if args.cmd is None:
        args = p.parse_args(["live"])
    try:
        args.func(args)
    except PyMongoError as e:
        sys.exit(f"MongoDB error: {e}")


if __name__ == "__main__":
    main()
