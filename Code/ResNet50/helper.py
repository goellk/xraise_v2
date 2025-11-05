import os
import json
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image

# Custom dataset class
class MixedRailwayDataset(Dataset):
    def __init__(self, img_dir, annot_dir, transform=None):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        annot_path = os.path.join(self.annot_dir, img_name.rsplit('.', 1)[0] + '.json')

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Load annotation
        with open(annot_path, 'r') as f:
            annotation = json.load(f)

        # Determine label (1 if "person" exists, otherwise 0)
        label = 1 if any(ann["class"] == "person" for ann in annotation.get("annotations", [])) else 0

        if self.transform:
            image = self.transform(image)

        return image, label


class VGG16_BinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16_BinaryClassifier, self).__init__()
        # Load VGG16 model
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        # Modify the classifier to output a single value (binary classification)
        self.vgg.classifier[6] = nn.Sequential(
            nn.Linear(self.vgg.classifier[6].in_features, 1),
            nn.Sigmoid()
        )
        # Use adaptive average pooling to handle varying image sizes
        self.vgg.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        return self.vgg(x)

class VGG11_BinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG11_BinaryClassifier, self).__init__()
        # Load VGG11 model
        self.vgg = models.vgg11(weights=models.VGG11_Weights.DEFAULT if pretrained else None)
        # Modify the classifier to output a single value (binary classification)
        self.vgg.classifier[6] = nn.Sequential(
            nn.Linear(self.vgg.classifier[6].in_features, 1),
            nn.Sigmoid()
        )
        # Use adaptive average pooling to handle varying image sizes
        self.vgg.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        return self.vgg(x)

# ResNet50 model for binary classification
class ResNet50_BinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_BinaryClassifier, self).__init__()
        # Load ResNet50 model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        # Add adaptive average pooling to handle varying image sizes
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Set output size to (1, 1)
        # Modify the fully connected layer to output a single value (binary classification)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)


# ResNet18 model for binary classification
class ResNet18_BinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18_BinaryClassifier, self).__init__()
        # Load ResNet18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Add adaptive average pooling to handle varying image sizes
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Set output size to (1, 1)
        # Modify the fully connected layer to output a single value (binary classification)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)


# ConvNeXt-Tiny model for binary classification
class ConvNextTiny_BinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ConvNextTiny_BinaryClassifier, self).__init__()
        # Load ConvNeXt-Tiny model
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        # Modify the classifier to output a single value (binary classification)
        self.convnext.classifier[2] = nn.Sequential(
            nn.Linear(self.convnext.classifier[2].in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.convnext(x)


