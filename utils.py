from torchvision import datasets, transforms
from torch.utils.data import DataLoader , Dataset
import torch
import os
from PIL import Image
import pandas as pd
import copy
import numpy as np

# NEW: for reports/plots
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# The dataset Class
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_folder = './data'


class ChestData (Dataset) :
    def __init__ (self, dataframe, img_dir , transform=None) :
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__ (self) :
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['X_ray_image_name']
        label = int(row['target'])  # ensure int
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------- dataframes ----------
def get_df () :
    df = pd.read_csv(os.path.join(data_folder, 'Chest_xray_Corona_Metadata.csv'))

    # Encode labels: Normal=0, Pneumonia=1
    df['target'] = df['Label'].map({'Normal': 0, 'Pnemonia': 1})

    # Split into train/test
    train_df_chest = df[df['Dataset_type'] == 'TRAIN'].copy()
    test_df_chest  = df[df['Dataset_type'] == 'TEST'].copy()
    return train_df_chest , test_df_chest


# ---------- dataloaders ----------
def get_dataloaders ( data_dir = data_folder , batch_size = 32, num_workers=0 ) :
    train_df_chest , test_df_chest = get_df()
    

# TRAIN transform: strong augmentation
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

# TEST transform: no augmentation
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
   # Datasets and Loaders (KEEPING YOUR RELATIVE PATHS)
    train_dataset_chest = ChestData(
        train_df_chest,
        os.path.join(data_dir,'Coronahack-Chest-XRay-Dataset','train'),
        train_transform
    )
    test_dataset_chest = ChestData(
        test_df_chest,
        os.path.join(data_dir,'Coronahack-Chest-XRay-Dataset','test'),
        test_transform
    )
   


    train_loader_chest = DataLoader(train_dataset_chest, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader_chest  = DataLoader(test_dataset_chest,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader_chest, test_loader_chest


def save_best_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)




# ---------- NEW: evaluation & saving helpers ----------
def evaluate_full(model, loader):
    """Return arrays: y_true, y_pred, y_prob on a given loader."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).float().unsqueeze(1)

            out = model(X)
            p = torch.sigmoid(out)              # prob for class=1
            pred = (p > 0.5).float()

            y_true.extend(y.squeeze(1).cpu().numpy().tolist())
            y_pred.extend(pred.squeeze(1).cpu().numpy().tolist())
            y_prob.extend(p.squeeze(1).cpu().numpy().tolist())
    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int), np.array(y_prob, dtype=float)


def save_curves(trainLoss, testLoss, trainAcc, testAcc, out_dir):
    """Save history.csv + loss_curve.png + acc_curve.png"""
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    hist_df = pd.DataFrame({
        "epoch": np.arange(1, len(trainLoss)+1),
        "train_loss": trainLoss.numpy(),
        "test_loss":  testLoss.numpy(),
        "train_acc":  trainAcc.numpy(),
        "test_acc":   testAcc.numpy(),
    })
    hist_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # Loss curve
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    plt.plot(hist_df["epoch"], hist_df["test_loss"],  label="Test Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss per Epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png")); plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_acc"], label="Train Acc")
    plt.plot(hist_df["epoch"], hist_df["test_acc"],  label="Test Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy per Epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curve.png")); plt.close()


def save_report_and_cm(y_true, y_pred, out_dir, target_names=("Normal","Pnemonia")):
    """Save classification_report.txt + confusion_matrix.png"""
    os.makedirs(out_dir, exist_ok=True)


    # text report
    report = classification_report(y_true, y_pred, target_names=list(target_names), digits=4)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)



    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(target_names))
    plt.xticks(ticks, target_names, rotation=45)
    plt.yticks(ticks, target_names)
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png")); plt.close()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ---------- training ----------
def trainTheModel(numepochs, optimizer, lossfun, model, train_loader, test_loader, scheduler=None,
                  save_dir=None, model_name=None):
  
    early_stopper = EarlyStopping(patience=5)

    # initialize losses and accuracies
    trainLoss = torch.zeros(numepochs)
    testLoss  = torch.zeros(numepochs)
    trainAcc  = torch.zeros(numepochs)
    testAcc   = torch.zeros(numepochs)
    model  = model.to(device)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"\n🚀 Starting training for **{model.__class__.__name__}** for {numepochs} epochs \n")

    for epochi in range(numepochs):
        # ----- TRAINING -----
        model.train()
        batchLoss = []
        batchAcc  = []

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device).float().unsqueeze(1)

            yHat = model(X)
            loss = lossfun(yHat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(yHat)
            preds = (probs > 0.5).float()
            batchLoss.append(loss.item())
            batchAcc.append((preds == y).float().mean().item())

        trainLoss[epochi] = np.mean(batchLoss)
        trainAcc[epochi]  = 100*np.mean(batchAcc)

        # ----- VALIDATION -----
        model.eval()
        batchLoss = []
        batchAcc  = []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device).float().unsqueeze(1)

                yHat = model(X)
                loss = lossfun(yHat, y)

                probs = torch.sigmoid(yHat)
                preds = (probs > 0.5).float()
                batchLoss.append(loss.item())
                batchAcc.append((preds == y).float().mean().item())

        testLoss[epochi] = np.mean(batchLoss)
        testAcc[epochi]  = 100*np.mean(batchAcc)

        # ----- TRACK BEST MODEL -----
        if testAcc[epochi] > best_acc:
            best_acc = testAcc[epochi].item()
            best_model_wts = copy.deepcopy(model.state_dict())
            save_best_model(model, os.path.join('models', f"best_{model.__class__.__name__}_model.pth"))

        # ----- SCHEDULER STEP -----
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(testLoss[epochi].item())
            else:
                scheduler.step()

        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epochi+1}/{numepochs} | "
              f"Train Acc: {trainAcc[epochi]:.2f}% | Test Acc: {testAcc[epochi]:.2f}% | "
              f"Train Loss: {trainLoss[epochi]:.4f} | Test Loss: {testLoss[epochi]:.4f} | "
              f"LR: {current_lr:.6f}")
        
        if early_stopper(testLoss[epochi]):
            print(f"Early stopping at epoch {epochi+1}")
            break

        

    # Load best weights before returning
    model.load_state_dict(best_model_wts)

    # ----- OPTIONAL: save curves + report/CM -----
    if save_dir is not None:
        # curves
        curves_dir = os.path.join(save_dir, "curves",f'{model.__class__.__name__}' )
        save_curves(trainLoss, testLoss, trainAcc, testAcc, curves_dir)

        # eval on test set
        y_true, y_pred, _ = evaluate_full(model, test_loader)
        reports_dir = os.path.join(save_dir, "reports", f'{model.__class__.__name__}')
        save_report_and_cm(y_true, y_pred, reports_dir)

        print(f"\n📈 Curves saved to: {curves_dir}")
        print(f"📝 Report & Confusion Matrix saved to: {reports_dir}\n")

    return trainLoss, testLoss, trainAcc, testAcc, model
