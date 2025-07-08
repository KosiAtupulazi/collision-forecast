import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# TODO: Import your model and dataset classes
# from src.model import build_your_model
# from src.dataloader import YourDatasetClass


def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10):
    """
    Generic training loop for regression tasks in PyTorch, with validation.
    Args:
        model: The PyTorch model to train
        train_loader: DataLoader for your training data
        val_loader: DataLoader for your validation data
        optimizer: Optimizer (e.g., Adam, SGD)
        criterion: Loss function (e.g., nn.MSELoss)
        device: torch.device("cuda") or torch.device("cpu")
        num_epochs: Number of training epochs
    """
    model.to(device)
    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        # --- Validation phase ---
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # No gradients needed for validation
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs).squeeze()
                v_loss = criterion(val_outputs, val_targets.float())
                val_loss += v_loss.item() * val_inputs.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
    # Save the trained model
    torch.save(model.state_dict(), "model_regression.pth")
    print("Model saved as model_regression.pth")


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: Update the following lines with your dataset and model
    # train_dataset = YourDatasetClass(csv_path="path/to/your_train.csv", split="train")
    # val_dataset = YourDatasetClass(csv_path="path/to/your_val.csv", split="val")
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    # model = build_your_model()
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10)
    pass  # Remove this after filling in the TODOs 