import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from mml.utils.kernel_functions import ExponentialKernel, RBFKernel
from mml.data import biovid
import mml.models.kernel_transformer as models
import os
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ROOT_DIR_DAT = "F:/Users/Furkan/Documents/Master Thesis/Artificial Emotional Intelligence/code/mml/datasets/BioVid/PartA"
CSV_FILE = os.path.join(ROOT_DIR_DAT, 'samples.csv')
dat_transform = biovid.ToTensor()
biovid_part = biovid.BioVid_PartA_bio(CSV_FILE, ROOT_DIR_DAT,classes=[0, 4],
                                      modalities='gsr', transform=dat_transform)

exponential_kernel = RBFKernel() #ExponentialKernel()
model_config = models.kernerl_transformer_config_medium
embed_config = models.embed_config_basic
model = models.KernelTransformerModel(True, model_config, embed_config)

optimizer = Adam(model.parameters(), lr=0.001)
criterion = BCEWithLogitsLoss()

epochs = 10
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    biovid_part.train()
    biovid_part.loso_split(epoch % 87 + 1)
    train_dataloader = DataLoader(biovid_part, batch_size=10, shuffle=True)
    
    epoch_loss = 0
    correct = 0
    total = 0

    start = time.time()
    for sample in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()
        
        gsr = sample['gsr']
        gsr = gsr.to(device)
        label = sample['label']
        label[label == 4] = 1
        label = label.float().unsqueeze(1)
        
        output = model(gsr)
        output = output.float()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        prediction = (output > 0.5).float()
        correct += (prediction == label).sum().item()
        total += label.size(0)
    train_loss = epoch_loss / len(train_dataloader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    end = time.time()

    model.eval()
    biovid_part.val()
    val_dataloader = DataLoader(biovid_part, batch_size=10, shuffle=False)
    
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for sample in tqdm(val_dataloader, desc="Validation"):
            gsr = sample['gsr']
            label = sample['label']
            label[label == 4] = 1
            label = label.float().unsqueeze(1)
            
            output = model(gsr)
            loss = criterion(output, label)
            
            val_loss += loss.item()
            prediction = (output > 0.5).float()
            val_correct += (prediction == label).sum().item()
            val_total += label.size(0)

    val_loss /= len(val_dataloader)
    val_accuracy = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    print(f"Time: {end-start:.2f}s")
    print("-" * 50)