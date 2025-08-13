# %%
import sys
import os

project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)
print(sys.path)

# %%
from model import build_videomae
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import CollisionForecastDataset

# %%
def train_model(model, optimizer, criterion, device_param, train_loader, val_loader, FORCE_RETRAIN, epochs=10):
    model_name = model.__class__.__name__.lower()
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")

    # Check if model already exists
    if not FORCE_RETRAIN and os.path.exists(model_path):
        print(f"Model {model_name} already exists. Skipping training.")
        model.load_state_dict(torch.load(model_path))
        model.to(device_param)
        #if we skip training, we still get a loaded, valid model back, not None
        return model

    # # Define loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device_param)
    best_val_loss = float('inf')

    # Early stopping setup
    patience = 3  # Stop if validation loss doesn't improve for 3 epochs
    no_improvement = 0
    
    
    for e in range(epochs):
        model.train()
        for batch_idx, (input, target) in enumerate (train_loader):
            print(f"Train Epoch {e}, Batch {batch_idx}, inputs shape {input.shape}, target shape {target.shape}")
            input = input.to(device_param) # clip_name = input
            target = target.to(device_param) # label = target

            optimizer.zero_grad() #clears grads after each batch
            #[batch, num_fames, num_channels, width, height]
            input = input.permute(0, 2, 1, 3, 4) #the order videomae expects
            outputs = model(input) # does a forward pass
            loss = criterion(outputs.logits.squeeze(), target) # calculates the loss b/w the o/p & target
            loss.backward() #calculates the gradients
            optimizer.step() #updates the weights

            if batch_idx % 10 == 0:
                print(f"Train Epoch {e}, Batch {batch_idx}, Loss {loss.item()}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch_idx, (val_input, val_target) in enumerate(val_loader):
                val_input = val_input.to(device_param)
                val_target = val_target.to(device_param)
                #[batch, num_fames, num_channels, width, height]
                val_input = val_input.permute(0, 2, 1, 3, 4) #the order videomae expects
                
                val_outputs = model(val_input)
                batch_loss = criterion(val_outputs.logits.squeeze(), val_target)
                val_loss += batch_loss.item()

        val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {val_loss}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            no_improvement += 1
            print(f"No improvement for {no_improvement} epochs")
        # Early stopping check
        if no_improvement >= patience:
            print(f"Early stopping after {e+1} epochs - no improvement for {patience} epochs")
            break
        # Save the model every 5 epochs
        if e % 5 == 0: 
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    return model


# %%
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if device == "cuda":
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     model = build_videomae()
#     train_dataset = CollisionForecastDataset(csv_path="/home/atupulazi/personal_projects/collision-forecast/labels/train_clip_labels.csv", split="train")
#     val_dataset = CollisionForecastDataset(csv_path="/home/atupulazi/personal_projects/collision-forecast/labels/val_clip_labels.csv", split="val" )

#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


#     trained_model = train_model(
#     model=model,
#     optimizer=optimizer,
#     criterion=criterion,
#     device_param=device,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     FORCE_RETRAIN=True,
#     epochs=num_epochs
# )



# # %%
#     for inputs, targets in train_loader:
#         print("inputs shape:", inputs.shape)
#         print("targets shape:", targets.shape)
#         break  

#     # %%
#     # 1. Collect all validation targets
#     all_targets = []
#     for _, targets in val_loader:
#         all_targets.append(targets)
#     all_targets = torch.cat(all_targets)  # shape: [num_samples]

#     # 2. Calculate the mean of all targets
#     mean_value = all_targets.float().mean()

#     # 3. Create baseline predictions (all the same value)
#     baseline_predictions = torch.full_like(all_targets, mean_value.item())
#     # 4. Calculate MSE loss
#     mse_loss = torch.nn.functional.mse_loss(baseline_predictions, all_targets.float())
#     print("Baseline MSE loss:", mse_loss.item())
