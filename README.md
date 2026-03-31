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

1. First of all, you need to place the dataset into the Data/ folder. The dataset folder should consist of two directories, "test" and "training". In these directories, there should be two subdirectories "imgs" and "annots". The images should be placed in the "imgs" directory, and the json annotation files in the "annots" directory. Images should be in .jpg or .png format, annotations in JSON format. An annotation file should have the same name as the corresponding image file (apart from the file format example_name.png/jpg -> example_name.json) and should look like this example:
```json
{
    "metadata": {
        "dataset_name": "OSDaR23",
        "original_image_name": "091_1631640361.200000048.png",
        "scene_name": "3_fire_site_3.4",
        "rgb_directory": "rgb_right"
    },
    "annotations": [
        {
            "bbox": {
                "x_min": 9.99581603245975,
                "y_min": 947.8472957618069,
                "x_max": 42.38581603245975,
                "y_max": 1052.847295761807
            },
            "class": "person"
        },
        {
            "bbox": {
                "x_min": 13.943741009983375,
                "y_min": 942.6896432224502,
                "x_max": 45.22374100998338,
                "y_max": 1058.0096432224502
            },
            "class": "person"
        }
    ]
}
```
If the image contains a person, the bounding box data should be provided as shown in the example above. If no person is in the image, then leave the "annotations" section empty.

2. Then you need to train a model (VGG16/ResNet50/ConvNeXt-t). Open the training.py script inside the corresponding model directory and change the dataset paths and hyperparameters in the header:
```python
# TRAINING SCRIPT FOR CONVNEXT-T MODEL

#################################################################################################
# SETUP START - Change parameters if necessary
#################################################################################################

# Relative paths to training and test splits (images and annotation files)
TRAINING_IMG_DIR = "/Data/CUSTOM_DATASET_v3_unified/training/imgs"
TRAINING_ANNOT_DIR = "/Data/CUSTOM_DATASET_v3_unified/training/annots"
TEST_IMG_DIR = "/Data/CUSTOM_DATASET_v3_unified/test/imgs"
TEST_ANNOT_DIR = "/Data/CUSTOM_DATASET_v3_unified/test/annots"

# Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
EPOCHS = 50

#################################################################################################
# SETUP END
#################################################################################################
```
Now you can start training by running
```bash
python training.py
```
After each epoch, the training script will automatically save the model checkpoint (so you can use it later for XAI evaluation). Additionally, the training and valdiation losses and accuracies are logged in a .csv file.
5. Finally you can run an XAI method (Grad-CAM/LRP/CRAFT/CRP) on a trained model. Therefore, you also need to adjust the paths first. Open the gradcam|lrp|craft|crp.py file and change the parameters in the header, e.g. from craft.py:

```python
# CRAFT FOR VGG16 MODEL

#################################################################################################
# SETUP START - Change parameters if necessary
#################################################################################################

# Relative path to model checkpoint
MODEL_PATH = "vgg_models_512_v3/vgg_training_512_epoch_1.pth"

# Relative path to dataset that should be evaluated with CRAFT
EVAL_IMGS = "/Data/CUSTOM_DATASET_v3_unified/test/imgs/"
EVAL_ANNOTS = "/Data/CUSTOM_DATASET_v3_unified/test/annots/"

# Relative path to dataset from which concept images should be taken from
CONCEPT_IMGS = "/Data/CUSTOM_DATASET_v3_unified/concept/imgs/"
CONCEPT_ANNOT = "/Data/CUSTOM_DATASET_v3_unified/concept/annots/"

# Flag to control whether to show the prediction text on the CRAFT attribution maps
SHOW_PREDICTION = True

#################################################################################################
# SETUP END
#################################################################################################
```
If you set the ```SHOW_PREDICTION``` flag to ```False```, only the explanation (i.e. heatmap) will be shown, without the classification output.
Run the XAI method by executing:
```bash
python lrp.py
```
   
*Please note that it is strongly recommended to a rather small concept image datapool (especially for CRAFT) because of the memory requirements - sticking to a maximum of ~300 images should work for 32GB of RAM.*

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
