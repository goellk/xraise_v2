#################################################################################################
# INFERENCE SCRIPT FOR VGG16 MODEL
#################################################################################################

import torch
from torchvision import transforms
from tqdm import tqdm
import csv
import os
from helper import MixedRailwayDataset, ResNet50_BinaryClassifier

#################################################################################################
# SETUP START - Change parameters if necessary
#################################################################################################

# Path to the trained model
MODEL_PATH = "resnet50_models_512_v3/resnet50_training_512_epoch_2.pth" 

# Relative paths to test split (images and annotation files)
TEST_IMG_DIR = "/Data/CUSTOM_DATASET_v3_unified/cropped_dataset/test/imgs"
TEST_ANNOT_DIR = "/Data/CUSTOM_DATASET_v3_unified/cropped_dataset/test/annots"

# Output log file
LOG_FILE = "inference_results_resnet.csv"

#################################################################################################
# SETUP END
#################################################################################################

# Define transformation
test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.40933493, 0.42142126, 0.41395313], 
                        std=[0.2761048, 0.28513926, 0.29439896])
])

# Get workspace directory
workspace_dir = str(os.path.dirname(os.path.dirname(os.getcwd())))
test_img_dir = workspace_dir + TEST_IMG_DIR
test_annot_dir = workspace_dir + TEST_ANNOT_DIR

# Prepare dataset
print("Preparing test dataset...")
test_dataset = MixedRailwayDataset(test_img_dir, test_annot_dir, transform=test_transform)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
print("Loading model...")
model = ResNet50_BinaryClassifier(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("Starting inference...")

# CSV logging
with open(LOG_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "sigmoid_output", "classification", "ground_truth"])

# Run inference
for idx in tqdm(range(len(test_dataset))):
    img_name = test_dataset.image_files[idx]
    
    # Get image and ground truth
    image, label = test_dataset[idx]
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Forward pass
    with torch.no_grad():
        output = model(image)
    
    sigmoid_out = output.item()
    pred_label = 1 if sigmoid_out > 0.5 else 0
    
    classification = "person" if pred_label == 1 else "not person"
    ground_truth = "person" if label == 1 else "no person"
    
    # Log result
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([img_name, f"{sigmoid_out:.6f}", classification, ground_truth])

print(f"Inference complete! Results saved to {LOG_FILE}")
