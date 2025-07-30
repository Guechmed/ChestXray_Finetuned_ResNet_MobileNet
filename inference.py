import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import os
import torch.nn as nn 
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transform: test-time (NO augmentation)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Paths (dataset CSV)
DATA_DIR = "./data"
CSV_PATH = os.path.join(DATA_DIR, "Chest_xray_Corona_Metadata.csv")

def load_model(model_path, model_type='mobilenet'):
    if model_type == 'mobilenet':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, 1)
        )
    else:  # ResNet18
        model = models.resnet18(pretrained=False)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 1)
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(image_path, model, threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()

    label = "Pneumonia" if prob > threshold else "Normal"
    return label, prob

def get_true_label(image_name):
    """Return true label from CSV if the file exists in metadata."""
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    row = df[df["X_ray_image_name"] == image_name]
    if len(row) == 0:
        return None
    return row.iloc[0]["Label"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "mobilenet"],
                        help="Which model to use: mobilenet or resnet")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--image", type=str, required=True, help="Path to image to predict")
    parser.add_argument("--threshold", type=float, default=0.7, help="Decision threshold")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image {args.image} not found")

    model = load_model(args.weights, model_type=args.model)
    label, prob = predict(args.image, model, args.threshold)

    # Get true label (if exists)
    image_name = os.path.basename(args.image)
    true_label = get_true_label(image_name)

    print(f"✅ Prediction: {label}  |  Probability of Pneumonia: {prob:.3f}  | Threshold: {args.threshold}")
    if true_label:
        print(f"✅ True Label (from CSV): {true_label}")
    else:
        print("⚠️ True label not found in metadata.")
