# app/streamlit_app.py
import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# local utils paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(ROOT, "data")
TEST_DIR = os.path.join(DATA_DIR, "Coronahack-Chest-XRay-Dataset", "test")
CSV_PATH = os.path.join(DATA_DIR, "Chest_xray_Corona_Metadata.csv")
MODELS_DIR = os.path.join(ROOT, "models")

LABEL2IDX = {"Normal": 0, "Pnemonia": 1}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

@st.cache_resource
def load_metadata():
    df = pd.read_csv(CSV_PATH)
    df = df[df["Label"].isin(LABEL2IDX.keys())].copy()
    df["target"] = df["Label"].map(LABEL2IDX).astype(int)
    # restrict to TEST rows that physically exist
    df = df[df["Dataset_type"] == "TEST"]
    df = df[df["X_ray_image_name"].apply(lambda n: os.path.exists(os.path.join(TEST_DIR, n)))]
    return df

def build_model(name):
    if name == "MobileNetV2":
        m = models.mobilenet_v2(pretrained=False)
        m.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),  # dropout for regularization
            nn.Linear(m.classifier[1].in_features, 1)   
            )
    else:  # ResNet18
        m = models.resnet18(pretrained=False)
        m.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.fc.in_features, 1)
        )
    return m

@st.cache_resource
def load_model(name: str):
    mdl = build_model(name)
    mdlf = "best_MobileNetV2_model.pth" if name == "MobileNetV2" else "best_ResNet_model.pth"
    model_path = os.path.join(MODELS_DIR, mdlf)
    mdl.load_state_dict(torch.load(model_path, map_location=device))
    mdl.to(device).eval()
    return mdl

def predict(img, model,threshold=0.7):
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        p = torch.sigmoid(out).item()
    
    pred_idx = int(p > threshold)
    
    return pred_idx, p
# --------------- UI ----------------
st.set_page_config(page_title="Chest X-ray Classification", layout="wide")
st.title("🫁 Chest X-ray Classification (Normal vs Pneumonia)")

meta = load_metadata()
test_files = sorted(meta["X_ray_image_name"].unique().tolist())

# Sidebar for controls
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose model", ["ResNet18","MobileNetV2"])
threshold =  st.slider("Decision Threshold  default = 0.7" ,0.4, 0.9, 0.7, 0.05)
src = st.sidebar.radio("Image source", ["Pick from test set", "Upload your own"])

# Main layout with two columns (image + result)
col_img, col_res = st.columns([1.2, 1])  # 1.2x space for image

if src == "Pick from test set":
    selected_fname = st.selectbox("Select test image", test_files)
    img_path = os.path.join(TEST_DIR, selected_fname)
    img = Image.open(img_path)
    row = meta[meta["X_ray_image_name"] == selected_fname].iloc[0]
    true_label = row["Label"]

    with col_img:
        st.image(img, caption=f"🖼️ {selected_fname}", width=400)  # fixed width instead of full width

    with col_res:
        if st.button("🔍 Predict", use_container_width=True):
            model = load_model(model_name)
            pred_idx, prob_pos = predict(img, model, threshold)
            pred_label = IDX2LABEL[pred_idx]

            # Better styled card with contrasting colors
            st.markdown(f"""
            <div style='padding:15px; border-radius:10px; background:#1e1e1e; text-align:center; color:white'>
                <h3 style='margin-bottom:10px;'> <span style='color:#4dabf7'>Prediction Result</span></h3>
                <p style='font-size:18px'><b>True Label:</b> 
                    <span style='color:{"#4caf50" if true_label=="Normal" else "#4169e1"}'>{true_label}</span>
                </p>
                <p style='font-size:18px'><b>Predicted Label:</b> 
                    <span style='color:{"#4caf50" if pred_label=="Normal" else "#4169e1"}'>{pred_label}</span>
                </p>
                <p style='font-size:16px'><b>P(Pneumonia):</b> {prob_pos:.3f}</p>
            </div>

            """, unsafe_allow_html=True)
            
else:
    file = st.file_uploader("Upload image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if file is not None:
        img = Image.open(file)

        with col_img:
            st.image(img, caption=f"🖼️ {file.name}", width=400)

        # Try to get true label if filename matches test set
        base = os.path.basename(file.name)
        true_label = None
        if base in test_files:
            row = meta[meta["X_ray_image_name"] == base].iloc[0]
            true_label = row["Label"]
            st.info(f"**True label detected:** {true_label}")
        else:
            st.warning("True label unknown (not in test set).")

        with col_res:
            if st.button("🔍 Predict", use_container_width=True):
                with st.spinner("🔄 Predicting..."):
                    model = load_model(model_name)
                    pred_idx, prob_pos = predict(img, model, threshold)
                    pred_label = IDX2LABEL[pred_idx]

                # Using Streamlit native components instead of HTML
                st.success("Prediction Result")
                if true_label is not None:
                    color = "green" if true_label == "Normal" else "blue"
                    st.markdown(f"**True Label:** :{color}[{true_label}]")
                
                pred_color = "green" if pred_label == "Normal" else "blue"
                st.markdown(f"**Predicted Label:** :{pred_color}[{pred_label}]")
                st.metric("P(Pneumonia)", f"{prob_pos:.3f}")