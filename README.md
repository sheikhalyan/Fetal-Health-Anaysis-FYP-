<div align="center">

# 🩺 Fetal Health Analysis System

### Deep Learning–Powered Fetal Biometric Measurement from Ultrasound Images

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_Lite-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20Research-lightgrey?style=flat-square)]()

<br/>

> Automated segmentation and measurement of fetal biometric parameters using deep learning — built as a Final Year Project in Computer Science.

<br/>

</div>

---

## What It Does

This system allows clinicians and researchers to upload fetal ultrasound images and automatically receive:

- **Segmentation masks** highlighting the region of interest
- **Contour & ellipse detection** overlaid on the original image
- **Biometric measurements** calculated from the detected region
- **Visual result plots** ready for review or reporting

No manual tracing. No guesswork. Just upload and analyze.

---

## Supported Measurements

| Parameter | Description | Status |
|-----------|-------------|--------|
| **FL** | Femur Length | ✅ Implemented |
| **AC** | Abdominal Circumference | ✅ Implemented |
| **BPD** | Biparietal Diameter | ✅ Implemented |
| **HC** | Head Circumference | ✅ Implemented |

---

## Supported Ultrasound Machines

| Machine | Model File |
|---------|------------|
| Voluson E6 | `VLE6_MODEL.tflite` |
| Voluson S10 | `VLS10_MODEL.tflite` |
| Voluson S8 | `VLS8_MODEL.tflite` |
| ALOKA | `ALOKA_MODEL.tflite` |

---

## System Workflow

```
Upload Ultrasound Image
        ↓
Image Preprocessing & Normalization
        ↓
TFLite Model Inference (Segmentation)
        ↓
OpenCV Contour / Ellipse Detection
        ↓
Biometric Measurement Calculation
        ↓
Visualization + Results Display
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Web Framework** | Flask |
| **Deep Learning** | TensorFlow / TensorFlow Lite |
| **Image Processing** | OpenCV, NumPy |
| **Visualization** | Matplotlib |
| **Production Server** | Waitress (WSGI) |
| **Language** | Python 3.11 |

---

## Project Structure

```
FYP-FINAL-Server-webapp/
│
├── App.py                          # Main Flask application
│
├── models/                         # TFLite segmentation models
│   ├── AC_MODEL.tflite
│   ├── BPD_MODEL.tflite
│   ├── VLE6_MODEL.tflite
│   ├── VLS10_MODEL.tflite
│   ├── VLS8_MODEL.tflite
│   └── ALOKA_MODEL.tflite
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── Ac_Bpd_result_plot_images/  # AC & BPD result plots
│   └── Femur_result_plot_images/   # FL result plots per machine
│
├── templates/                      # Jinja2 HTML templates
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

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### 1. Clone the repository

```bash
https://github.com/sheikhalyan/Fetal-Health-Anaysis-FYP-.git
cd FYP-FINAL-Server-webapp
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
python App.py
```

**Production (Waitress)**
```python
from waitress import serve
serve(app, host="0.0.0.0", port=5000)
```

Then open your browser at:

```
http://127.0.0.1:5000
```

To access from another device on the same network:

```
http://YOUR_LOCAL_IP:5000
```

---

## Deployment

| Platform | Supported | Notes |
|----------|-----------|-------|
| Render | ✅ | Recommended |
| Railway | ✅ | Recommended |
| VPS / EC2 | ✅ | Full control |
| Vercel | ❌ | Serverless — not suitable |
| Netlify | ❌ | Serverless — not suitable |

> TensorFlow, OpenCV, and long-running Flask processes require a persistent server environment. Serverless platforms will not work.

---

## ⚠️ Disclaimer

This system was developed for **academic and research purposes** as part of a Final Year Project.

It is **not a certified medical diagnostic tool** and should not be used as a substitute for professional clinical judgment or certified medical equipment.

---

## Author

**Alyan**
*Computer Science Graduate*

[![GitHub](https://img.shields.io/badge/GitHub-@sheikhalyan-181717?style=flat-square&logo=github)](https://github.com/sheikhalyan)

*Interests: Deep Learning · Medical Imaging · Data Science · Web Development*

---

## Acknowledgements

- [TensorFlow](https://tensorflow.org) — Model inference engine
- [OpenCV](https://opencv.org) — Image processing and contour detection
- [Flask](https://flask.palletsprojects.com) — Web framework
- Academic supervisors and mentors
- Medical imaging research community

---

<div align="center">

Built with ❤️ as a Final Year Project in Computer Science

*For questions or feedback, please [open an issue](https://github.com/sheikhalyan/FYP-FINAL-Server-webapp/issues).*

</div>
