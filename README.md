# XRAISE — Explainable AI for Railway Safety Evaluations

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![Project Status](https://img.shields.io/badge/status-active-brightgreen.svg)

**XRAISE** (Explainable AI for Railway Safety Evaluations) is a research project by the **Deutsches Zentrum für Schienenverkehrsforschung (DZSF)** at the **Eisenbahn-Bundesamt**.  
This repository contains the source code for training convolutional neural networks on a custom dataset and applying various explainable AI (XAI) methods to support safety evaluations in the railway domain.

---

## 📘 Overview

This repository provides:

- Training code for three convolutional neural network architectures:
  - **VGG16**
  - **ResNet50**
  - **ConvNeXt-T**
- Implementations of four XAI methods:
  - **Grad-CAM**
  - **LRP (Layer-wise Relevance Propagation)**
  - **CRAFT**
  - **CRP (Contextual Relevance Propagation)**
- A `requirements.txt` file for setting up the Python environment.

---

## 📁 Repository Structure

```
XRAISE/
│
├── Code/
│   ├── VGG16/
│   │   ├── training.py
│   │   ├── helper.py
│   │   └── [XAI method scripts]
│   │
│   ├── ResNet50/
│   │   ├── training.py
│   │   ├── helper.py
│   │   └── [XAI method scripts]
│   │
│   ├── ConvNeXt-T/
│       ├── training.py
│       ├── helper.py
│       └── [XAI method scripts]
│
├── Data/
│   └── [datasets and preprocessing files]
│
└── requirements.txt
```

---

## 🧩 Environment Setup

To reproduce the experiments and run the code, we recommend using **Python 3.11** in a virtual environment.

### 1. Create and activate a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Instructions

*(To be completed — usage examples and run commands will be added later.)*

### Currently working XAI/model-combinations:
| Model       | Grad-CAM | LRP    | CRAFT | CRP |
|--------------|-------- |--------|-----------|-----|
| VGG16        |   ✅   |        | ✅        |     |
| ResNet50     |   ✅   |  ❌  | ✅        |     |
| ConvNeXt-T   |   ✅   |        | ✅        |     |

---

## 🧠 Citation

If you use this repository or parts of it in your work, please cite the **XRAISE** project appropriately.

---

## 📄 Acknowledgment

This work is part of the **XRAISE** research project by the  
**Deutsches Zentrum für Schienenverkehrsforschung (DZSF)** at the **Eisenbahn-Bundesamt**.

---
