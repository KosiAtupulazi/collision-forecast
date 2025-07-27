import torch
import json
import numpy as np
from model import build_videomae
from dataloader import CollisionForecastDataset
from torch.utils.data import DataLoader
import torch.nn as nn

# Load best hyperparameters
with open("best_hyperparams.json", "r") as f:
    best = json.load(f)

# Build and load model
model = build_videomae()
print('Type of model:', type(model))
print('Type of model.videomae:', type(model.videomae))
if not hasattr(model, 'videomae') or 'VideoMAEModel' not in str(type(model.videomae)):
    raise RuntimeError('model.videomae is not a VideoMAEModel. Check for variable shadowing or incorrect model instantiation.')
model.load_state_dict(torch.load("best_model.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # type: ignore
model.eval()

# Load test set
test_dataset = CollisionForecastDataset(csv_path="labels/test_clip_labels.csv", split="test")
test_loader = DataLoader(test_dataset, batch_size=best['batch_size'], shuffle=False)

# --- Register hook to capture attention ---
attention_maps = []
def hook_fn(module, input, output):
    # output is (attn_output, attn_weights) for HuggingFace models
    # attn_weights: [batch, num_heads, num_tokens, num_tokens]
    attention_maps.append(output[1].detach().cpu())

# Correct path for last self-attention layer
handle = model.videomae.encoder.layer[-1].attention.attention.register_forward_hook(hook_fn)

all_patch_attentions = []
all_confidences = []

# Evaluate
criterion = nn.MSELoss()
test_loss = 0
with torch.no_grad():
    for test_input, test_target in test_loader:
        test_input = test_input.to(device)
        test_target = test_target.to(device)
        test_input = test_input.permute(0, 2, 1, 3, 4)
        outputs = model(test_input)
        confidences = torch.sigmoid(outputs.logits).cpu().numpy()
        all_confidences.extend(confidences)
        # attention_maps[-1]: [batch, num_heads, num_tokens, num_tokens]
        attn_map = attention_maps[-1]  # shape: [batch, num_heads, num_tokens, num_tokens]
        for i in range(attn_map.shape[0]):
            # Average over heads and mean over source tokens (see attention_to_frame_weights)
            attn_avg = attn_map[i].mean(dim=0).mean(dim=0)  # shape: [num_tokens]
            all_patch_attentions.append(attn_avg.numpy())
        attention_maps.clear()
handle.remove()

np.save("patch_attentions.npy", np.stack(all_patch_attentions))
np.save("confidences.npy", np.array(all_confidences))

test_loss = test_loss / len(test_loader)
print(f"Test Loss: {test_loss}")