import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from pytorch_grad_cam import GradCAM
from helper import VGG16_BinaryClassifier  # Adjust if using another model
from tqdm import tqdm
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


def apply_gradcam(model, target_layer, image_path, output_dir):
    model.eval()
    model.to(device)

    # Load image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4045, 0.4143, 0.4043], std=[0.2852, 0.2935, 0.2993])
    ])
    tensor_img = transform(image).unsqueeze(0).to(device)

    # Get model prediction
    with torch.no_grad():
        output = model(tensor_img).item()
    predicted_class = "person" if output > 0.5 else "no person"
    probability = output * 100

    # GradCAM setup
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor_img)[0, :]

    # Convert to heatmap (no resizing needed, as grayscale_cam is already 512x512)
    heatmap = np.uint8(255 * grayscale_cam)  # Convert to uint8
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert PIL image to OpenCV format
    img_cv = np.array(image.resize((512, 512)))  # Resize the original image to 512x512
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # Convert RGB -> BGR

    # Blend heatmap and original image
    blended = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)

    if predicted_class == "person":
        rect_color = (0, 255, 0)  # Green rectangle
        text_color = (0, 0, 0)  # Black text
    else:
        rect_color = (255, 0, 0)  # Red rectangle
        text_color = (0,0,0)  # Black text

    # Convert back to PIL for annotation
    blended_pil = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(blended_pil)
    font = ImageFont.load_default(size=30)
    text = f"Predicted: {predicted_class} | p={probability:.2f}%"
    text_size = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]
    draw.rectangle([(10, 10), (10 + text_width + 10, 10 + text_height + 10)], fill=rect_color)
    draw.text((15, 15), text, fill=text_color, font=font)

    # Save result
    filename, ext = os.path.splitext(os.path.basename(image_path))
    output_path = os.path.join(output_dir, f"{filename}_gradcam{ext}")

    blended_pil.save(output_path)


# Paths and device
workspace_dir = str(os.path.dirname(os.path.dirname(os.getcwd())))
image_dir = workspace_dir + "/03_Original images_48/imgs/"


output_dir = "gradcam_results"
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = VGG16_BinaryClassifier(pretrained=False)
#model.load_state_dict(torch.load("vgg_models_512_v3/vgg_training_512_epoch_3.pth", map_location=device))
model.load_state_dict(torch.load("finetuned_models_512_inria_7/vgg16_512_finetuned_epoch_3.pth", map_location=device))
#model.load_state_dict(torch.load("vgg_inria_pretrained_models_512/vgg_inria_pretrained_512_epoch_7.pth", map_location=device))

model.to(device)

# Select target layer for Grad-CAM (adjust based on model architecture)
target_layer = model.vgg.features[-1]

# Process each image in directory
for image_file in tqdm(os.listdir(image_dir)):
    if image_file.endswith(('.jpg', '.png')):
        image_path = os.path.join(image_dir, image_file)
        apply_gradcam(model, target_layer, image_path, output_dir)


print("Grad-CAM visualization completed!")

