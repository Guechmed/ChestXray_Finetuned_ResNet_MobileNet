import os
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight

from utils import get_dataloaders, trainTheModel, save_best_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_folder = './data'

def build_model(model_name):
    if model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
        for p in model.parameters(): 
            p.requires_grad = False 
     
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),  # dropout for regularization
            nn.Linear(model.classifier[1].in_features, 1)   
            )


    elif model_name == "resnet":

        model = models.resnet18(pretrained=True)
        for p in model.parameters(): 
            p.requires_grad = False 

        model.fc = nn.Sequential(
            nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 1)
            
        )
        
    else:
        raise ValueError("Model must be 'mobilenet' or 'resnet'")
    return model


if __name__ == "__main__":
    # ---- Command-line arguments ----
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 or ResNet18 on Chest X-ray dataset")
    parser.add_argument("--model", type=str, default="mobilenet", help="Model name: mobilenet or resnet")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--weighted", action="store_true", help="Use class-weighted loss")
    parser.add_argument("--save_path", type=str, default="models", help="Where to save the best model")
    parser.add_argument("--output_path", type=str, default="output", help="Where to save the best model")
    parser.add_argument("--model_name", type=str, default="mobilenet", help="Model name: mobilenet or resnet")
    args = parser.parse_args()

    # ---- Load dataloaders ----
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # ---- Class-weight logic ----
    if args.weighted:
        df = pd.read_csv(os.path.join(data_folder, 'Chest_xray_Corona_Metadata.csv'))
        df['target'] = df['Label'].map({'Normal': 0, 'Pnemonia': 1})
        classes = np.unique(df['target'])
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=df['target'].values)
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        lossfun = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        print(f"✅ Using class-weighted loss. Weights: {class_weights.cpu().numpy()}")
    else:
        lossfun = nn.BCEWithLogitsLoss()

    # ---- Build model ----
    model = build_model(args.model).to(device)

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    

    # ---- Train ----
    trainLoss, testLoss, trainAcc, testAcc, best_model = trainLoss, testLoss, trainAcc, testAcc, best_model = trainTheModel(
    numepochs=args.epochs,
    optimizer=optimizer,
    lossfun=lossfun,
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    scheduler=scheduler,
    save_dir="outputs",        
           
)

    # ---- Save best model ----
    save_path = f"{args.save_path}/best_{model.__class__.__name__}.pth"
    save_best_model(best_model, save_path)
    print(f"✅ Best {args.model} model saved at {save_path}")

