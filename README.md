# 🫁 Chest X-ray Classification (Normal vs Pneumonia)

![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-blue?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Deployed%20UI-Streamlit-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

This project fine-tunes **MobileNetV2** and **ResNet18** CNN models to classify **chest X-rays** into **Normal** or **Pneumonia** categories using the [CoronaHack-Chest-XRay dataset](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset).

It includes:  
✅ Training scripts (with class balancing & data augmentation)  
✅ Evaluation tools (classification reports, confusion matrices, threshold tuning)  
✅ Inference script (predict single/multiple images from terminal)  
✅ Interactive Streamlit app (upload/test images with models)  

---

##  Project Overview

- **Transfer Learning:** fine-tuned `MobileNetV2` and `ResNet18`
- **Dataset:** `Chest_xray_Corona_Metadata.csv` + X-ray images
- Balanced class distribution using `class_weight` in loss function
- **Data augmentation:** flips, rotations, crops, color jitter for training  
- **Test-time transforms:** only resize & normalization (deterministic)
- Saved best models based on validation accuracy
- **Inference CLI**: predict images from the terminal with adjustable threshold  
- **Streamlit UI**: upload images and interactively test models

---


## 📈 Evaluation Results
### 🔹 Accuracy Curve
![Accuracy Curve](outputs/curves/ResNet/acc_curve.png)

### 🔹 Confusion Matrix
![Confusion Matrix](outputs/reports/ResNet/confusion_matrix.png)


### 🔹 Classification Report
```

              precision    recall  f1-score   support

      Normal     0.8502    0.8974    0.8732       234
    Pnemonia     0.9363    0.9051    0.9205       390
    accuracy                         0.9022       624
   macro avg     0.8933    0.9013    0.8968       624
weighted avg     0.9040    0.9022    0.9027       624
```

## 📂 Repository Structure
```
project/
├── app/
│ └── app.py # Interactive Streamlit app
│
├── data/
│   └── Coronahack-Chest-XRay-Dataset/
│   |    ├── train/          
│   |    └── test/
|   |___Chest_xray_Corona_Metadata.csv 
├── models/ # Saved model weights
│ ├── best_MobileNetV2_model.pth
│ └── best_ResNet_model.pth
├── outputs/ # Training curves, reports, confusion matrices
├── utils.py # Data loading & training utilities
├── train.py # CLI script for training models
├── inference.py # CLI script for predictions
├── requirements.txt # Project dependencies
└── README.md
```

## 🚀 How to Run

### -1 Install dependencies

```
python3 -m venv venv
source venv/bin/activate      
# (on Windows: venv\Scripts\activate)

pip install -r requirements.txt
```

### 2- Download the dataset
```    
data/
├── Coronahack-Chest-XRay-Dataset/
│   ├── train/
│   └── test/
└── Chest_xray_Corona_Metadata.csv
```

### 3  - 📊 Train Models

```  
----Train ResNet18 (with class-weighted loss) ----:

### 🔹 Train ResNet18 
```bash 
python train.py --model resnet --epochs 50 --weighted

----Train MobileNetV2---- : 
python train.py --model mobilenet --epochs 50 --weighted

U can pass oter arguments to the command too :

python train.py --model resnet --epochs 50 --lr 1e-4 --batch_size 64  --weighted --save_path models --output_path output 

=========
Models are saved automatically under models/

Accuracy & loss curves are saved to outputs/
====== 
``` 
###  4- 🔍 Run Inference (Terminal)
```

python inference.py \
    --model resnet \
    --weights models/best_MobileNetV2_model.pth \
    --image data/Coronahack-Chest-XRay-Dataset/test/IM-0001-0001.jpeg \
    --threshold 0.7

Output :  
✅ Prediction: Pneumonia  |  Probability of Pneumonia: 0.732  | Threshold: 0.7
🎯 True Label (from CSV): Pneumonia

``` 

### 5- 🌐 Run Streamlit App
```
streamlit run app/app.py
``` 





