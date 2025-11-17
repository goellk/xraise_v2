import numpy as np
import torch
import cv2
from torchvision import transforms as T
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import warnings
from helper import *
from craft.craft_torch import Craft, torch_to_numpy
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import colorsys
from PIL import Image, ImageDraw, ImageFont
import os

warnings.filterwarnings("ignore")

# Set variables
workspace_dir = str(os.path.dirname(os.path.dirname(os.getcwd())))
MODEL_PATH = "resnet50_models_512/resnet50_training_512_epoch_3.pth"
#IMAGE_DIR = workspace_dir + "/03_Original images_48/imgs/"
#ANNOT_DIR = workspace_dir + "/03_Original images_48/annots/"

# Images for inference with CRAFT
IMAGE_DIR = workspace_dir + "/08_Data/CRAFT_MAPS_DATASET/imgs/"
ANNOT_DIR = workspace_dir + "/08_Data/CRAFT_MAPS_DATASET/annots/"

# Concept images
CONCEPT_IMAGE_DIR = workspace_dir + "/08_Data/CRAFT_CONCEPT_DATASET+INRIA/imgs/"
CONCEPT_ANNOT_DIR = workspace_dir + "/08_Data/CRAFT_CONCEPT_DATASET+INRIA/annots/"

OUTPUT_PATH = "craft_results/"
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH + "concept_attribution_maps/", exist_ok=True)
os.makedirs(OUTPUT_PATH + "concepts/", exist_ok=True)

# Define image preprocessing
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.4045, 0.4143, 0.4043], std=[0.2852, 0.2935, 0.2993])
])

# Load model
def load_model(model_path):
    model = ResNet50_BinaryClassifier(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

print("Loading model...")
model = load_model(MODEL_PATH)

# Processing
#config = resolve_data_config({}, model=model)
#config["input_size"] = (3, 512, 512)
#transform = create_transform(**config)
to_pil = T.ToPILImage()

# Load dataset
dataset = MixedRailwayDataset(IMAGE_DIR, ANNOT_DIR, transform=transform)
concept_dataset = MixedRailwayDataset(CONCEPT_IMAGE_DIR, CONCEPT_ANNOT_DIR, transform=transform)

image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
concept_image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]

print(f"Total dataset size: {len(dataset)}")
if len(dataset) == 0:
    raise ValueError("Dataset is empty! Check IMAGE_DIR and ANNOT_DIR paths.")

print(f"Total concept-dataset size: {len(concept_dataset)}")
if len(concept_dataset) == 0:
    raise ValueError("Concept-dataset is empty! Check CONCEPT_IMAGE_DIR and CONCEPT_ANNOT_DIR paths.")


# Extract original image names
original_image_names = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]

# Process images
print("Processing images...")
images_preprocessed_all = torch.stack([dataset[i][0] for i in range(len(dataset))])
print(f"Loaded and processed {images_preprocessed_all.size(0)} images for attribution maps.")
images_preprocessed_filtered = torch.stack([concept_dataset[i][0] for i in range(len(concept_dataset))])
print(f"Loaded and processed {images_preprocessed_filtered.size(0)} images for concepts.")
print("Finished image processing!")

# Cut the model in two
print("Cutting model...")
g = nn.Sequential(*list(model.resnet.children())[:-2])  # Excludes avgpool and fully connected
h = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),  # Reapply avgpool
    nn.Flatten(),  # Flatten output for fully connected layer
    model.resnet.fc  # Fully connected layer
)
print("Cutting model done.")

# Initialize CRAFT
craft = Craft(input_to_latent=g,
              latent_to_logit=h,
              number_of_concepts=10,
              patch_size=64,
              batch_size=BATCH_SIZE,
              device=DEVICE)

print("Fitting CRAFT on filtered dataset...")
crops, crops_u, w = craft.fit(images_preprocessed_filtered)
crops = np.moveaxis(torch_to_numpy(crops), 1, -1)

# Compute importance
print("Computing importance scores for all images...")
importances = craft.estimate_importance(images_preprocessed_all, class_id=0)
images_u = craft.transform(images_preprocessed_all)

# Most important concepts
print("Getting most important concepts...")
most_important_concepts = np.argsort(importances)[::-1][:5]
for c_id in most_important_concepts:
  print("Concept", c_id, " has an importance value of ", importances[c_id])

nb_crops = 9
def show(img, **kwargs):
    img = np.array(img)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img -= img.min();img /= img.max()
    plt.imshow(img, **kwargs); plt.axis('off')

for c_id in most_important_concepts:
    best_crops_ids = np.argsort(crops_u[:, c_id])[::-1][:nb_crops]
    best_crops = crops[best_crops_ids]

    print("Concept", c_id, " has an importance value of ", importances[c_id])
    for i in range(nb_crops):
        plt.subplot(ceil(nb_crops/5), 5, i+1)
        show(best_crops[i])


def get_alpha_cmap(cmap):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        c = np.array((cmap[0]/255.0, cmap[1]/255.0, cmap[2]/255.0))
        cmax = colorsys.rgb_to_hls(*c)
        cmax = np.array(cmax)
        cmax[-1] = 1.0
        cmax = np.clip(np.array(colorsys.hls_to_rgb(*cmax)), 0, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [c,cmax])
    alpha_cmap = cmap(np.arange(256))
    alpha_cmap[:,-1] = np.linspace(0, 0.85, 256)
    alpha_cmap = ListedColormap(alpha_cmap)
    return alpha_cmap

cmaps = [
  get_alpha_cmap((54, 197, 240)),
  get_alpha_cmap((210, 40, 95)),
  get_alpha_cmap((236, 178, 46)),
  get_alpha_cmap((15, 157, 88)),
  get_alpha_cmap((84, 25, 85))
]


def plot_legend():
    for concept_idx, c_id in enumerate(most_important_concepts):
        plt.figure(figsize=(6, 6))  # Set square figure size for 3x3 grid

        cmap = cmaps[concept_idx % len(cmaps)]  # Get colormap
        best_crops_ids = np.argsort(crops_u[:, c_id])[::-1][:9]  # Get top 9 crops
        best_crops = crops[best_crops_ids]

        for i in range(9):
            row = i % 3  # 3 rows
            col = i // 3  # 3 columns

            plt.subplot(3, 3, i + 1)  # Arrange in 3x3 grid
            show(best_crops[i])

            # Add bounding box
            rect = plt.Rectangle((0, 0), best_crops[i].shape[1], best_crops[i].shape[0],
                                 linewidth=3, edgecolor=cmap(1.0), facecolor='none')
            plt.gca().add_patch(rect)
            plt.axis('off')

        # Add title with concept importance
        plt.suptitle(f"Concept {c_id}\nImportance: {importances[c_id]:.2f}",
                     fontsize=14, color=cmap(1.0), fontweight='bold')

        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + "concepts/" + "concept" + str(c_id) + ".jpg")


def plot_concept_attribution_maps(percentile=90):
    for id in range(len(images_preprocessed_all)):
        img = images_preprocessed_all[id]
        u = images_u[id]

        # Convert the preprocessed image back to a displayable format
        img_np = torch_to_numpy(img).transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize to [0, 1]

        # Get model prediction
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(DEVICE)).item()
        predicted_class = "person" if output > 0.5 else "no person"
        probability = output * 100

        # Create a new figure for the original image + all concept heatmaps overlay
        plt.figure(figsize=(6, 6))

        # Display the original image
        plt.imshow(img_np)

        # Overlay each concept heatmap on the original image
        for i, c_id in enumerate(most_important_concepts):
            cmap = cmaps[i]
            heatmap = u[:, :, c_id]

            # Only show concept if it exceeds the N-th percentile
            sigma = np.percentile(images_u[:, :, :, c_id].flatten(), percentile)
            heatmap = heatmap * np.array(heatmap > sigma, np.float32)

            # Resize heatmap to match the original image size
            heatmap = cv2.resize(heatmap[:, :, None], (img_np.shape[1], img_np.shape[0]))

            # Overlay the heatmap on the original image
            plt.imshow(heatmap, cmap=cmap, alpha=0.5)  # Adjust alpha for transparency

        # Add prediction text and rectangle
        if predicted_class == "person":
            rect_color = (0, 255, 0)  # Green rectangle
            text_color = (0, 0, 0)  # Black text
        else:
            rect_color = (255, 0, 0)  # Red rectangle
            text_color = (0, 0, 0)  # Black text

        # Convert the plot to a PIL image for annotation
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("temp.png",bbox_inches='tight', pad_inches=0)
        plt.close()

        # Open the saved image and add the prediction text
        blended_pil = Image.open("temp.png")
        blended_pil = blended_pil.resize((512, 512), Image.Resampling.LANCZOS)

        draw = ImageDraw.Draw(blended_pil)
        font = ImageFont.load_default(size=30)
        text = f"Predicted: {predicted_class} | p={probability:.2f}%"
        text_size = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]
        draw.rectangle([(10, 10), (10 + text_width + 10, 10 + text_height + 10)], fill=rect_color)
        draw.text((15, 15), text, fill=text_color, font=font)

        # After adding the text and rectangle, resize the image to 512x512
        # Save the final image with the prediction text
        original_image_name = original_image_names[id]
        blended_pil.convert("RGB").save(OUTPUT_PATH + "concept_attribution_maps/" + f"{original_image_name}_craft.jpg")
        #os.remove("temp.png")  # Remove the temporary file

plot_legend()
plot_concept_attribution_maps()

