import torch
from torchvision import transforms
from helper import *
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.image import imgify, vis_opaque_img, plot_grid
from crp.visualization import FeatureVisualization
import warnings
import numpy as np
import cv2
from PIL import ImageDraw, ImageFont
from tqdm import tqdm


matplotlib.use('Agg')  # Use non-interactive backend to prevent pop-ups
warnings.filterwarnings("ignore")  # Ignore warnings

# Input directories
workspace_dir = str(os.path.dirname(os.path.dirname(os.getcwd())))
heatmap_img_dir = workspace_dir + "/03_Original images_48/imgs/"
concept_img_dir = workspace_dir + "/08_Data/CRP_CONCEPT_DATASET/imgs/"
concept_annot_dir = workspace_dir + "/08_Data/CRP_CONCEPT_DATASET/annots/"

# Output directory
output_dir = "crp_results"
os.makedirs(output_dir, exist_ok=True)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VGG16_BinaryClassifier(pretrained=False)
model.load_state_dict(torch.load("finetuned_models_512_inria_7/vgg16_512_finetuned_epoch_3.pth", map_location=device))
#model.load_state_dict(torch.load("vgg_models_512_train/vgg_training_512_epoch_5.pth", map_location=device))
model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4045, 0.4143, 0.4043], std=[0.2852, 0.2935, 0.2993])
])

# Prepare CRP
cc = ChannelConcept()
composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
attribution = CondAttribution(model, no_param_grad=True)
layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])

# Function for running inference on given image and save the classification result
def run_inference(image_path, output_dir):
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((512, 512))
    tensor_img = transform(image_resized).unsqueeze(0).to(device)

    # Get model prediction
    with torch.no_grad():
        output = model(tensor_img).item()
    predicted_class = "person" if output > 0.5 else "no person"
    probability = output * 100

    # Convert PIL image to OpenCV format (keeping 512x512 size)
    img_cv = np.array(image_resized)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Define text box color and text color
    rect_color = (0, 255, 0) if predicted_class == "person" else (255, 0, 0)
    text_color = (0, 0, 0)

    # Convert back to PIL for annotation
    annotated_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(annotated_pil)
    font = ImageFont.load_default(size=30)
    text = f"Predicted: {predicted_class} | p={probability:.2f}%"
    text_size = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]
    draw.rectangle([(10, 10), (10 + text_width + 10, 10 + text_height + 10)], fill=rect_color)
    draw.text((15, 15), text, fill=text_color, font=font)

    # Save result
    filename, ext = os.path.splitext(os.path.basename(image_path))
    output_path = os.path.join(output_dir, f"{filename}_inference{ext}")
    annotated_pil.save(output_path)


# Loop over all images and run inference
print("Running inference...")
for img_file in tqdm(os.listdir(heatmap_img_dir)):
    img_path = os.path.join(heatmap_img_dir, img_file)

    # Create output folder for image
    img_output_dir = os.path.join(output_dir, os.path.splitext(img_file)[0]+"_crp")
    os.makedirs(img_output_dir, exist_ok=True)

    # Run inference on image
    run_inference(img_path, img_output_dir)


# Loop over all images in heatmap_img_dir and run CRP
print("Performing CRP...")
for img_file in os.listdir(heatmap_img_dir):
    img_path = os.path.join(heatmap_img_dir, img_file)
    # Create output folder for image
    img_output_dir = os.path.join(output_dir, os.path.splitext(img_file)[0] + "_crp")
    os.makedirs(img_output_dir, exist_ok=True)

    # Load and process image
    image = Image.open(img_path)
    sample = transform(image).unsqueeze(0).to(device)
    sample.requires_grad = True

    # Compute attribution
    conditions = [{"y": [0]}]
    attr = attribution(sample, conditions, composite, record_layer=layer_names)
    rel_c = cc.attribute(attr.relevances['vgg.features.28'], abs_norm=True)

    # Get top k most relevant concepts
    rel_values, concept_ids = torch.topk(rel_c[0], 5)
    print(f"{img_file} - Concepts: {concept_ids.tolist()}, Relevance: {(rel_values*100).tolist()}")

    # Create heatmaps for each concept_id
    for id, rel_value in zip(concept_ids, rel_values):
        condition = [{'vgg.features.28': [id], 'y': [0]}]
        heatmap, _, _, _ = attribution(sample, condition, composite)
        heatmap_img = imgify(heatmap, symmetric=True, grid=(1, 1))

        # Convert heatmap to a numpy array if it's not already
        if isinstance(heatmap_img, torch.Tensor):
            heatmap_img = heatmap_img.squeeze().cpu().numpy()

        # Create and save matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(heatmap_img, cmap='viridis', vmin=0, vmax=1)
        ax.axis('off')
        plt.suptitle(
            f"Concept {id}\nRelevance: {rel_value*100:.2f}%", fontsize=14, color='black', fontweight='bold', y=0.95)
        plt.savefig(os.path.join(img_output_dir, f"{os.path.splitext(img_file)[0]}_heatmap_{id}.png"),
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)  # Close the figure to free memory


    ### Concepts visualization
    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonPlusFlat(canonizers)
    layer_map = {layer : cc for layer in layer_names}
    attribution = CondAttribution(model)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    preprocessing =  transforms.Normalize(mean=[0.4045, 0.4143, 0.4043], std=[0.2852, 0.2935, 0.2993])
    dataset = MixedRailwayDataset(concept_img_dir, concept_annot_dir, transform=transform)
    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=preprocessing,
                              path=img_output_dir+"/FeatureVisualization/")
    saved_files = fv.run(composite, 0, len(dataset), 4, 100)

    # Iterate over each concept and its images
    ref_c = fv.get_max_reference(concept_ids.tolist(), "vgg.features.28",
                                 "relevance", (0, 9), rf=True, composite=composite, plot_fn=vis_opaque_img)
    for concept_id, images in ref_c.items():
        rel_score = rel_values[concept_ids.tolist().index(concept_id)] * 100  # Get the relevance score

        # Create a 3x3 plot for the 9 images
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        #fig.suptitle(f"Concept {concept_id}, \nImportance: {rel_score:.2f}%", fontsize=14, fontweight='bold')
        fig.suptitle(f"Concept {concept_id}", fontsize=14, fontweight='bold')

        for i, ax in enumerate(axes.flat):
            if i < len(images):
                ax.imshow(images[i])
                ax.axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(img_output_dir, f"{os.path.splitext(img_file)[0]}_concept_{concept_id}_grid.png"))
        plt.close(fig)

    # Delete FeatureVisualization directory (not needed for further steps)
    shutil.rmtree(img_output_dir + "/FeatureVisualization/")

print("Processing complete!")
