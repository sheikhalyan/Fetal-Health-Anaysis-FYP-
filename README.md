# 🩺 Fetal Health Analysis System

### Deep Learning–Powered Automated Fetal Biometric Measurement from Ultrasound Images

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_Lite-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20Research-lightgrey?style=flat-square)](LICENSE)

> > A multi-model deep learning system for the automated segmentation and biometric measurement of fetal anatomical structures from ultrasound images — built as a Final Year Research Project at PAF-KIET, with a research paper currently not published.

**yet to be Published research paper:** *"Multi-Model Fetal Biometry: An Integrated Approach using V-Net Model for automated segmentation and measurement of Femur Length, Abdominal Circumference, and Biparietal Diameter with Head Circumference in Ultrasound Images"*

---

## What It Does

Upload a fetal ultrasound image, select the measurement type and machine model, and the system automatically returns:

- **Segmentation mask** highlighting the anatomical region of interest
- **Ellipse/contour overlay** fitted to the predicted mask
- **Quantitative biometric measurements** (in mm) — no manual tracing required
- **Side-by-side visual comparison** of original image and predicted result

---

## Model Performance

Trained on **3,703 ultrasound images** across four machine types, evaluated using Dice Similarity Coefficient (DSC), Mean IoU, and Hausdorff Distance (HD).

### AC, HC & BPD Models

| Model | DSC (%) | Mean IoU (%) | HD (mm) | Precision (%) | Recall (%) | AUC (%) |
|-------|---------|--------------|---------|---------------|------------|---------|
| **AC** | 95.88 | 92.09 | 7.00 | 93.23 | 98.69 | 99.14 |
| **BPD / HC** | **98.45** | **96.95** | **4.47** | 97.54 | 99.38 | 99.58 |

### Femur Length Models (per Ultrasound Machine)

| Machine | DSC (%) | Mean IoU (%) | Accuracy (%) | Training Epochs |
|---------|---------|--------------|--------------|-----------------|
| Voluson E6 | 92.93 | 86.79 | 99.95 | 120 |
| Voluson S10 | 87.38 | 77.59 | 99.88 | 200 |
| Aloka | 86.95 | 76.91 | 99.88 | 120 |
| Voluson S8 | 81.36 | 68.58 | 99.87 | 120 |

### Comparison with State-of-the-Art (HC/BPD)

Our BPD/HC model achieves a DSC of **98.45%**, competitive with published state-of-the-art methods on the same benchmark:

| Method | Year | DSC (%) | HD (mm) |
|--------|------|---------|---------|
| SAF-Net (Liu et al.) | 2020 | 98.05 | 1.27 |
| Mask R-CNN (Al-Bander et al.) | 2019 | 97.73 | 1.39 |
| Random Forest (Heuvel et al.) | 2018 | 97.10 | 1.83 |
| Mini Link-Net (Sobhaninia et al.) | 2019 | 96.84 | 1.72 |
| GVF-Net (Rong et al.) | 2019 | 95.53 | 2.18 |
| **Our BPD/HC Model (V-Net)** | **2024** | **98.45** | **4.47** |

---

## Supported Measurements & Machines

| Parameter | Description | Status |
|-----------|-------------|--------|
| **FL** | Femur Length | ✅ Implemented |
| **AC** | Abdominal Circumference | ✅ Implemented |
| **BPD** | Biparietal Diameter | ✅ Implemented |
| **HC** | Head Circumference | ✅ Implemented |

| Ultrasound Machine | Model File | Dataset Size |
|--------------------|------------|--------------|
| Voluson E6 | `VLE6_MODEL.tflite` | 633 images |
| Aloka | `ALOKA_MODEL.tflite` | 294 images |
| Voluson S10 | `VLS10_MODEL.tflite` | 74 images |
| Voluson S8 | `VLS8_MODEL.tflite` | 48 images |

---

## System Architecture

```
Upload Ultrasound Image
        ↓
Select Measurement Type + Machine
        ↓
Image Preprocessing & Normalization (388×664×1)
        ↓
V-Net Model Inference (TFLite segmentation)
        ↓
Binary Mask Generation (probability threshold)
        ↓
OpenCV Contour / Ellipse Detection
        ↓
Biometric Measurement Calculation (mm)
        ↓
Visualization + Results Display
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Web Framework** | Flask 2.x |
| **Deep Learning** | TensorFlow / TensorFlow Lite |
| **Model Architecture** | V-Net (adapted for 2D segmentation) |
| **Image Processing** | OpenCV, NumPy, Matplotlib |
| **Production Server** | Waitress (WSGI) |
| **Training Platform** | Google Colab (Tesla T4 GPU) |
| **Language** | Python 3.11 |

---

## Dataset

- **Femur images:** 1,038 images from [Zenodo](https://zenodo.org/record/3904280) across 4 machine types
- **AC images:** 1,386 images
- **HC images:** 1,279 images
- **Data split:** 80% training / 20% evaluation
- **Masks:** Manually annotated in collaboration with gynecologist Dr. Talat Naz

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/sheikhalyan/fetal-health-analysis.git
cd fetal-health-analysis
```

### 2. Create and activate a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

**Development**
```bash
python app.py
```

**Production (Waitress)**
```python
from waitress import serve
serve(app, host="0.0.0.0", port=5000)
```

Then open: `http://127.0.0.1:5000`

---

## Deployment

| Platform | Supported | Notes |
|----------|-----------|-------|
| Render | ✅ | Recommended |
| Railway | ✅ | Recommended |
| VPS / EC2 | ✅ | Full control |
| Vercel | ❌ | Serverless — not suitable |
| Netlify | ❌ | Serverless — not suitable |

> TensorFlow, OpenCV, and long-running Flask processes require a **persistent server environment**. Serverless platforms will not work.

---

## Project Structure

```
fetal-health-analysis/
│
├── app.py                              # Main Flask application
│
├── models/                             # TFLite segmentation models
│   ├── AC_MODEL.tflite
│   ├── BPD_MODEL.tflite
│   ├── VLE6_MODEL.tflite
│   ├── VLS10_MODEL.tflite
│   ├── VLS8_MODEL.tflite
│   └── ALOKA_MODEL.tflite
│
├── static/
│   ├── css/style.css
│   ├── Ac_Bpd_result_plot_images/
│   └── Femur_result_plot_images/
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── femur.html
│   ├── result.html
│   └── result_femur.html
│
├── requirements.txt
└── README.md
```

---

## Research Paper

This project is based on original research conducted as part of the Final Year Project:

**"Multi-Model Fetal Biometry: An Integrated Approach using V-Net Model"**
Sheikh Alyan, Syed Saqib, Areeb Ahmed, Abdul Samad
PAF-KIET, 2024

Key contributions:
- First unified framework covering all four major fetal biometric parameters (FL, AC, BPD, HC) in a single system
- Multi-machine support: models trained separately per ultrasound machine type for higher accuracy
- BPD/HC model achieves 98.45% DSC, competitive with published state-of-the-art

---

## ⚠️ Disclaimer

This system was developed for **academic and research purposes** as a Final Year Project.

It is **not a certified medical diagnostic tool** and must not be used as a substitute for professional clinical judgment or certified medical equipment.

---

## Author

**Sheikh Alyan** — BS Computer Science, PAF-KIET (2020–2024)


[![GitHub](https://img.shields.io/badge/GitHub-@sheikhalyan-181717?style=flat-square&logo=github)](https://github.com/sheikhalyan)

---

## Acknowledgements

- [TensorFlow](https://tensorflow.org) — Model inference engine
- [OpenCV](https://opencv.org) — Image processing and contour detection
- [Flask](https://flask.palletsprojects.com) — Web framework
- Dr. Talat Naz (Gynecologist) — Expert annotation support
- [Zenodo Dataset](https://zenodo.org/record/3904280) — Femur ultrasound images
- Academic supervisors and the medical imaging research community
