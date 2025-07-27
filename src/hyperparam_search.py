import itertools
import torch
from model import build_videomae
from dataloader import CollisionForecastDataset
from torch.utils.data import DataLoader
from train import train_model
import torch.nn as nn
import torch.optim as optim
import json



# -------------------------
# Grid (defaults)
# -------------------------
learning_rates = [1e-3, 3e-4, 1e-4]          
batch_sizes    = [8, 16]
optimizers     = ['AdamW', 'Adam', 'RMSprop', 'SGD']    
weight_decays  = [1e-4, 1e-5, 0.0]
epochs         = [20]                        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()

results = []
best_val_loss = float('inf')
best_model_state = None
best_hyperparams = None

# Build datasets once; loaders depend on batch_size so we build inside loop
# train_dataset = CollisionForecastDataset(csv_path="labels/train_clip_labels.csv", split="train")
# val_dataset   = CollisionForecastDataset(csv_path="labels/val_clip_labels.csv",   split="val")

train_dataset = CollisionForecastDataset(csv_path="/home/atupulazi/personal_projects/collision-forecast/labels/train_clip_labels.csv", split="train")
val_dataset = CollisionForecastDataset(csv_path="/home/atupulazi/personal_projects/collision-forecast/labels/val_clip_labels.csv", split="val" )

for lr, batch_size, optimizer_name, weight_decay, num_epochs in itertools.product(
        learning_rates, batch_sizes, optimizers, weight_decays, epochs):

    print(f"\n==> lr={lr}, bs={batch_size}, opt={optimizer_name}, wd={weight_decay}, epochs={num_epochs}")

    model = build_videomae()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # Optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Train
    trained_model = train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device_param=device,
        train_loader=train_loader,
        val_loader=val_loader,
        FORCE_RETRAIN=True,
        epochs=num_epochs
    )

    # Validate
    trained_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_input, val_target in val_loader:
            val_input  = val_input.to(device).permute(0, 2, 1, 3, 4)
            val_target = val_target.to(device)
            outputs = trained_model(val_input).logits.squeeze()
            batch_loss = criterion(outputs, val_target)
            val_loss += batch_loss.item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.6f}")

    trial = {
        'lr': lr,
        'batch_size': batch_size,
        'optimizer': optimizer_name,
        'weight_decay': weight_decay,
        'epochs': num_epochs,
        'val_loss': val_loss
    }
    results.append(trial)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = trained_model.state_dict()
        best_hyperparams = trial

    
print("\nAll results:")
for r in results:
    print(r)

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    with open("best_hyperparams.json", "w") as f:
        json.dump(best_hyperparams, f, indent=2)
    print("\nSaved best_model.pth and best_hyperparams.json")

print("\nBest hyperparameters:")
print(json.dumps(best_hyperparams, indent=2))
