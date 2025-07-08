from src.model import build_videomae
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataloader import CollisionForecastDataset

def train_model(model, lr, device, train_loader, val_loader, epochs=10):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    model.train()
    for e in range(epochs):
    


if __name__ == "__main__":
    train_dataset = CollisionForecastDataset(csv_path="labels/train_clip_labels.csv", split="train")
    val_dataset = CollisionForecastDataset(csv_path="labels/val_clip_labels.csv", split="val" )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

