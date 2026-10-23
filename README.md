# LumenS--A-Spectra-Analyzer 🌊🔬
<div align="center">

### A DIY, camera-based spectrometer for screening impurities in water 💧

*Turn a webcam, a diffraction grating and some Python into a tool that helps you see what's hiding in your water.*

![Python](https://img.shields.io/badge/Python-3.9%2B-2E8B57?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-1E90FF?style=for-the-badge&logo=opencv&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Storage-3FA037?style=for-the-badge&logo=mongodb&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Math-00A8A8?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Peaks-4CAF50?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-0B7285?style=for-the-badge)
![ML](https://img.shields.io/badge/ML-Impurity%20Classifier-8BC34A?style=for-the-badge)
![Made for](https://img.shields.io/badge/Made%20for-Clean%20Water-00B4D8?style=for-the-badge)

🌱 &nbsp; 💧 &nbsp; 🌍 &nbsp; ♻️ &nbsp; 🔬 &nbsp; 🌿 &nbsp; 🐟 &nbsp; 🌊

</div>

---

## 🌍 Why This Project?

Clean water is one of the most basic needs on the planet, yet testing it usually needs expensive lab equipment. **AquaSpectra** is a low-cost, open approach: a webcam captures the light spectrum passing through (or emitted by) a water sample, the software calibrates it to real wavelengths, and the spectral fingerprint is stored so that **machine learning** can learn to recognise the signatures of common impurities such as:

| 🧪 Impurity family | 💧 Examples | 🌿 Why it matters |
|---|---|---|
| 🪙 **Heavy / trace metals** | Copper, iron, lead, chromium | Toxic at low concentrations, often colour water |
| 🧂 **Dissolved salts** | Chlorides, sulfates, nitrates | Affect taste, farming and drinking safety |
| 🟤 **Organics & turbidity** | Tannins, algae, suspended matter | Signs of pollution or contamination |

> ⚠️ **Disclaimer:** AquaSpectra is an educational / screening tool. Results are *indicative* and are **not** a substitute for certified laboratory testing of drinking water.

---

## 🌟 Features

- 📷 **Live camera capture** with ROI (Region of Interest) selection over the spectrum band
- 🎞️ **Frame averaging** to reduce sensor noise (default 10 frames)
- 🌈 **Wavelength calibration**: click known spectral lines, enter their nm values, get a fitted pixel → nm curve (linear for 2 points, quadratic for 3+) with RMS error reported
- 📈 **Per-channel spectra**: Red, Green, Blue and averaged Intensity
- 🏔️ **Automatic peak detection** (smoothed, prominence-based, up to 8 peaks)
- ⚠️ **Saturation warning** when the sensor is overexposed (≥ 250/255)
- 🗄️ **MongoDB storage** of every spectrum, calibration and its metadata
- 🔍 **CLI tools** to list, plot, compare, export (CSV) and delete saved spectra
- 🛟 **Graceful fallback**: runs without a database if MongoDB is unreachable

---

## 🧰 Apps, Libraries & Tools Used

### 🐍 Software

| 🌿 Tool | 🎯 Purpose in this project |
|---|---|
| **Python 3** | Core language |
| **OpenCV (`cv2`)** | Webcam capture, ROI selection, live preview windows, colour-space conversion |
| **NumPy** | Array maths, channel averaging, polynomial calibration fitting (`polyfit` / `polyval`) |
| **SciPy (`scipy.signal.find_peaks`)** | Detecting spectral peaks |
| **Matplotlib** | Spectrum plots, interactive calibration clicking (`ginput`), comparison overlays |
| **MongoDB** | Database holding the `spectra` and `calibrations` collections |
| **PyMongo** (incl. `bson`) | Python driver for MongoDB, `ObjectId` handling, indexes |
| **argparse** *(stdlib)* | Command-line interface |
| **csv** *(stdlib)* | Exporting spectra to CSV |
| **datetime / os / sys** *(stdlib)* | Timestamps (UTC), environment config, clean exits |

### 🔧 Hardware (typical DIY setup)

- 📷 USB webcam
- 🌈 Diffraction grating or slit assembly inside a light-tight enclosure
- 💡 Light source (e.g. white LED) and a transparent cuvette / sample holder for the water
- 🔦 A calibration source with known lines, for example a **CFL bulb**

### 🤖 Machine Learning layer *(roadmap)*

The current code collects and structures the data that ML needs (calibrated wavelengths, intensity, per-channel values, peaks, sample labels). The classifier itself is the next stage:

- 🧠 **scikit-learn** (planned) for baseline models such as Random Forest / SVM / PLS regression
- 🌲 Features: calibrated intensity curve, peak positions and heights, channel ratios
- 🎯 Targets: impurity type (metal / salt / clean) and, later, approximate concentration

---

## 🗺️ How It Works

```
 💧 Water sample
      │  light passes through
      ▼
 🌈 Diffraction grating  ──►  📷 Webcam
                                  │
                                  ▼
                   🟩 ROI selected + 10 frames averaged
                                  │
                                  ▼
            📐 Pixel → Wavelength calibration (nm)
                                  │
                                  ▼
          🏔️ Peak detection  +  R/G/B/Intensity curves
                                  │
                                  ▼
                       🗄️ MongoDB (spectra DB)
                                  │
                                  ▼
                  🤖 ML model → impurity prediction
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repo
```bash
git clone https://github.com/akshk-19/<your-repo-name>.git
cd <your-repo-name>
```

### 2️⃣ Install dependencies 🌿
```bash
pip install opencv-python numpy matplotlib scipy pymongo
```
> 💡 `bson` ships with PyMongo. Do **not** install the separate `bson` package from PyPI, as it conflicts.

### 3️⃣ Start MongoDB 🍃
Run a local instance (default `mongodb://localhost:27017`) or point to MongoDB Atlas.

### 4️⃣ Configure *(optional)* ⚙️

| Variable | Default |
|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` |
| `MONGO_DB` | `spectrometer` |

```bash
export MONGO_URI="mongodb://localhost:27017"
export MONGO_DB="spectrometer"
```

### 5️⃣ Launch 🔬
```bash
python spectrometer.py live --camera 1 --avg 10
```
Running `python spectrometer.py` with no arguments also starts live mode.

---

## 🎮 Live-View Controls

| Key | Action |
|:---:|---|
| 🟩 **r** | Select ROI: drag a box over the spectrum band, press ENTER |
| 🌈 **c** | Calibrate: click known spectral lines, then type their wavelengths |
| 📸 **s** | Capture spectrum, preview it, optionally save to MongoDB |
| 🛑 **q** | Quit |

### 🌈 Quick calibration guide
1. Shine your calibration source (e.g. CFL) into the spectrometer.
2. Press **r** and select the spectrum band.
3. Press **c**, click at least **2** known lines left to right, press ENTER.
4. Type the wavelength (nm) for each point and name the calibration.
5. The new calibration becomes **active** and is reused automatically next time.

> 🎯 More calibration points means a better fit. With 3+ points a quadratic is used, and the RMS error in nm is printed.

---

## 💻 CLI Commands

```bash
python spectrometer.py live [--camera 1] [--avg 10]   # 📷 live capture
python spectrometer.py list [--sample NAME] [--limit 20]  # 📋 list spectra
python spectrometer.py plot <ID>                      # 📈 plot one spectrum
python spectrometer.py compare <ID1> <ID2> ...        # 🔀 overlay (normalized)
python spectrometer.py export <ID> -o water.csv       # 💾 export to CSV
python spectrometer.py delete <ID>                    # 🗑️ delete a spectrum
```

---

## 🗄️ Data Model

**`spectra` collection** 🌊

| Field | Description |
|---|---|
| `sample`, `notes` | Label (e.g. "tap water", "CuSO4 5 ppm") and free text |
| `created_at` | UTC timestamp |
| `camera_index`, `frames_averaged`, `roi` | Capture settings |
| `pixel_x`, `wavelengths` | Pixel axis and calibrated nm axis (`null` if uncalibrated) |
| `intensity`, `r`, `g`, `b` | Spectral curves |
| `peaks`, `peaks_unit` | Detected peaks (nm or px) |
| `calibration_id` | Link to the calibration used |
| `saturated` | Overexposure flag |
| `legacy_hue_wavelength_nm` | Old hue-based estimate from the first project, kept for reference only |

**`calibrations` collection** 📐: name, pixels, wavelengths, polynomial `coeffs`, `degree`, `rms_error_nm`, `active`, `created_at`.

---

## 🌱 Tips for Good Measurements

- 🕶️ Keep the enclosure **light-tight**; stray light ruins spectra.
- ☀️ Avoid saturation: lower the light or camera exposure if you see the warning.
- 🧪 Always record a **reference spectrum of pure / distilled water** first.
- 🔁 Keep the same ROI, camera index and lighting between samples.
- 🏷️ Label samples consistently; good labels make better ML training data.

---

## 🛣️ Roadmap

- [x] Camera capture with frame averaging
- [x] Wavelength calibration and storage
- [x] Peak detection and MongoDB persistence
- [x] CSV export and comparison plots
- [ ] 🧪 Reference subtraction (sample vs. distilled water / absorbance)
- [ ] 🤖 ML classifier for metals vs. salts vs. clean water
- [ ] 📊 Concentration estimation (ppm) with calibration curves
- [ ] 🌐 Dashboard for browsing spectra
- [ ] 📱 Portable / Raspberry Pi field version

---

## 🤝 Contributing

Ideas, spectra datasets and pull requests are very welcome! 🌿 Fork the repo, create a branch, and open a PR.

---

## 👩‍💻 Author

**Akshita Kushwaha**
🔗 [GitHub](https://github.com/akshk-19) · [LinkedIn](https://linkedin.com/in/akshita-kushwaha-swe2006)

---

<div align="center">

### 🌍 *Every drop counts. Let's make water testing open to everyone.* 💧

🌱 🌊 🌿 ♻️ 🐠 🌎 💚

</div>
