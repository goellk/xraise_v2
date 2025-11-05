import csv
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from helper import *
import random
import numpy as np

# Reproducability
#os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # Set environment variable to enable deterministic algorithms
#torch.manual_seed(42)
#random.seed(42)
#np.random.seed(42)
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True, warn_only=True)
#def seed_worker(worker_id):
#    worker_seed = torch.initial_seed() % 2**32
#    np.random.seed(worker_seed)
#    random.seed(worker_seed)
#g = torch.Generator()
#g.manual_seed(42)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.40933493, 0.42142126, 0.41395313], std=[0.2761048, 0.28513926, 0.29439896])
])

# Paths to dataset directories
workspace_dir = str(os.path.dirname(os.path.dirname(os.getcwd())))
train_img_dir = workspace_dir + "/Data/CUSTOM_DATASET_v2_unified/training/imgs"
train_annot_dir = workspace_dir + "/Data/CUSTOM_DATASET_v2_unified/training/annots"

val_img_dir = workspace_dir + "/Data/CUSTOM_DATASET_v2_unified/test/imgs"
val_annot_dir = workspace_dir + "/Data/CUSTOM_DATASET_v2_unified/test/annots"

# Define hyperparameters
BATCH_SIZE = 4 #4
LEARNING_RATE = 0.0001
EPOCHS = 50

# Data preparation
print("Preparing data...")
train_dataset = MixedRailwayDataset(train_img_dir, train_annot_dir, transform=transform)
val_dataset = MixedRailwayDataset(val_img_dir, val_annot_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)#, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)#, worker_init_fn=seed_worker, generator=g)

# Initialize custom VGG16 model
print("Preparing model...")
model = ResNet50_BinaryClassifier(pretrained=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
print("Starting training...")
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Create a directory to save models
os.makedirs("resnet50_models_512", exist_ok=True)

# CSV logging
log_file = "log_resnet50_512.csv"
with open(log_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Epoch", "Train Loss", "Train Acc", "Train Prec", "Train Rec", "Train F1", "Val Loss", "Val Acc", "Val Prec",
         "Val Rec", "Val F1"])

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch [{epoch + 1}/{EPOCHS}]:")
    model.train()
    train_loss = 0.0
    train_labels, train_preds = [], []

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predictions = (outputs > 0.5).float()
        train_labels.extend(labels.cpu().numpy())
        train_preds.extend(predictions.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)
    train_prec = precision_score(train_labels, train_preds, zero_division=1)
    train_rec = recall_score(train_labels, train_preds, zero_division=1)
    train_f1 = f1_score(train_labels, train_preds, zero_division=1)
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_labels, val_preds = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predictions = (outputs > 0.5).float()
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(predictions.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_prec = precision_score(val_labels, val_preds, zero_division=1)
    val_rec = recall_score(val_labels, val_preds, zero_division=1)
    val_f1 = f1_score(val_labels, val_preds, zero_division=1)
    val_loss /= len(val_loader)

    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [epoch + 1, train_loss, train_acc, train_prec, train_rec, train_f1, val_loss, val_acc, val_prec, val_rec,
             val_f1])

    # Save the model after each epoch
    model_path = f"resnet50_models_512/resnet50_training_512_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

print("Training complete!")
