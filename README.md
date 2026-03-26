# XRAISE — Explainable AI for Railway Safety Evaluations

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![Project Status](https://img.shields.io/badge/status-active-brightgreen.svg)

**XRAISE** (Explainable AI for Railway Safety Evaluations) is a research project by the **Deutsches Zentrum für Schienenverkehrsforschung (DZSF)** at the **Eisenbahn-Bundesamt**.  
This repository contains the source code for training convolutional neural networks on a custom dataset and applying various explainable AI (XAI) methods to support safety evaluations in the railway domain.

---

## 📘 Overview

This repository provides:

- Training code for three convolutional neural network architectures:
  - **VGG16** [Simonyan & Zisserman, 2015]
  - **ResNet50** [He et al., 2016]
  - **ConvNeXt-T** [Liu et al., 2022]
- Implementations of four XAI methods:
  - **Grad-CAM (Gradient-weighted Class Activation Mapping)** [Selvaraju et al., 2017]
  - **LRP (Layer-wise Relevance Propagation)** [Bach et al., 2015]
  - **CRAFT (Concept Recursive Activation FacTorization)** [Fel et al., 2015]
  - **CRP (Concept Relevance Propagation)** [Achtibat et al., 2015]
- A `requirements.txt` file for setting up the Python environment.

Please note that you have to add datasets to the "Data" directory by yourself.

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
│   └── [datasets]
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
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Instructions

1. First of all, you need to place the dataset into the dataset folder. The dataset folder should consist of two directories, "test" and "training". In these directories, there should be two subdirectories "imgs" and "annots". The images should be placed in the "imgs" directory, and the json annotation files in the "annots" directory.
3. Then you need to train a model (VGG16/ResNet50/ConvNeXt-t).
4. Finally you can run an XAI method (Grad-CAM/LRP/CRAFT/CRP) on a trained model.
```bash
python lrp.py
```
   
*(To be completed — usage examples and run commands will be added later.)*

### Currently working XAI/model-combinations:
| Model       | Grad-CAM | LRP    | CRAFT | CRP |
|--------------|-------- |--------|-----------|-----|
| VGG16        |   ✅   |   ✅     | ✅      |  ✅   |
| ResNet50     |   ✅   |  ❌  | ✅        |  ✅    |
| ConvNeXt-T   |   ✅   |   ❌  | ✅        |   ❌  |

---

## 🧠 Citation

If you use this repository or parts of it in your work, please cite this github repository appropriately.

---

## 📚 References

### Model Architectures

- VGG16: Simonyan, K. & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition.
- ResNet: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
- ConvNeXt: Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s.

### XAI Methods

- Grad-CAM: Selvaraju, Ramprasaath R., et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.” 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618–626.
- LRP: Sebastian Bach, Alexander Binder, Grégoire Montavon, Frederick Klauschen, Klaus-Robert Müller, and Wojciech Samek. “On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation.” PLoS ONE, vol. 10, no. 7, 2015, e0130140.
- CRAFT: Thomas Fel, Agustin Picard, Louis Bethune, Thibaut Boissin, David Vigouroux, Julien Colin, Rémi Cadène, and Thomas Serre. “CRAFT: Concept Recursive Activation Factori­zation for Explainability.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
- CRP: Reduan Achtibat, Maximilian Dreyer, Ilona Eisenbraun, Sebastian Bosse, Thomas Wiegand, Wojciech Samek, and Sebastian Lapuschkin. “From Attribution Maps to Human‐Understandable Explanations through Concept Relevance Propagation.” Nature Machine Intelligence, vol. 5, no. 9, 2023, pp. 1006–1019.

---

## 📄 Acknowledgment
This work is part of the **XRAISE** research project by the  
**Deutsches Zentrum für Schienenverkehrsforschung (DZSF)** at the **Eisenbahn-Bundesamt**.

---
