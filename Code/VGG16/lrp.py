import os
import matplotlib
import torch
import numpy as np
from PIL import Image
from helper import VGG16_BinaryClassifier
from torchvision import transforms
from captum.attr import LRP
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, PropagationRule
import torch.nn as nn
import warnings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")

# Prevent plots from showing up during runtime
matplotlib.use('Agg')

# Configuration
workspace_dir = str(os.path.dirname(os.path.dirname(os.getcwd())))
TEST_IMAGE_DIR = workspace_dir + "/03_Original images_48/imgs/"


MODEL_PATH = "finetuned_models_512_inria_7/vgg16_512_finetuned_epoch_3.pth"
OUTPUT_DIR = "lrp_results"
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define same normalization as used in training
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4045, 0.4143, 0.4043], std=[0.2852, 0.2935, 0.2993]
    )
])


def load_binary_model(model_path):
    model = VGG16_BinaryClassifier(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_image_paths(image_dir):
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def apply_lrp_rules(model):
    # Apply LRP rules to the model layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.activation.Sigmoid):
            module.rule = EpsilonRule(0.001)
            print(f"Ignoring Sigmoid layer with zero epsilon: {name}")
            continue
        elif isinstance(module, nn.Linear) or isinstance(module, nn.AvgPool2d):
            module.rule = GammaRule(0.25)
        elif isinstance(module, nn.ReLU):
            module.rule = EpsilonRule(0.005)
        elif isinstance(module, nn.Conv2d):
            module.rule = EpsilonRule(0.05)
        else:
            print(f"Warning: No rule defined for layer {name} of type {type(module)}")


def lrp_analysis(image_paths, model, output_dir):
    # Apply LRP rules to the model
    results = []

    for img_path in tqdm(image_paths):
        try:
            model.eval()
            apply_lrp_rules(model)
            lrp = LRP(model)

            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')

            # Store the original image tensor before normalization
            original_img_tensor = transforms.Resize((512, 512))(img)
            original_img_tensor = transforms.ToTensor()(original_img_tensor).unsqueeze(0).to(DEVICE)

            # Apply normalization for the model input
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            # Forward pass
            with torch.no_grad():
                pred = model(img_tensor).item()

            # Determine predicted class
            predicted_class = "person" if pred > 0.5 else "no person"
            probability = pred * 100

            # Compute LRP attributions for person class (target=0 for binary)
            attribution = lrp.attribute(img_tensor, target=0)
            attribution = attribution.squeeze().cpu().detach().numpy()

            # Transpose for visualization
            attribution = np.transpose(attribution, (1, 2, 0))
            img_np = np.transpose(original_img_tensor.squeeze().cpu().numpy(), (1, 2, 0))  # Use original image tensor

            # Visualize
            fig, ax = plt.subplots(figsize=(10, 8))
            fig, ax = viz.visualize_image_attr_multiple(
                attribution,
                img_np,
                ["original_image", "heat_map"],
                ["all", "all"],  # Show both positive and negative relevances
                show_colorbar=True,
                outlier_perc=2,
                titles=["Original", "LRP Heatmap"],
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', ['red', 'white', 'green']),
                # Custom color map
                # signs="all"  # Show both positive and negative values
            )

            # Add prediction text to the plot
            if predicted_class == "person":
                rect_color = (0, 1, 0)  # Green rectangle
                text_color = (0, 0, 0)  # Black text
            else:
                rect_color = (1, 0, 0)  # Red rectangle
                text_color = (0, 0, 0)  # Black text

            # Add text annotation
            text = f"Predicted: {predicted_class} | p={probability:.2f}%"
            ax[0].text(10, 30, text, fontsize=12, color=text_color, bbox=dict(facecolor=rect_color, alpha=0.8))

            # Save results
            filename = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_dir, f"{filename}_lrp.jpg")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

            # Store results
            results.append({
                "image_path": img_path,
                "prediction": pred,
                "lrp_path": output_path
            })

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    # Save CSV report
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "lrp_results.csv"), index=False)
    return df


if __name__ == "__main__":
    # Load model
    model = load_binary_model(MODEL_PATH)
    print("Model loaded successfully")

    # Get test images
    image_paths = get_image_paths(TEST_IMAGE_DIR)
    print(f"Found {len(image_paths)} images for analysis")

    # Run LRP analysis
    results = lrp_analysis(image_paths, model, OUTPUT_DIR)
    print("LRP analysis completed. Results saved to:", OUTPUT_DIR)