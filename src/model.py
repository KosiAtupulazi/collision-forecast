import torch
import torch.nn as nn
import torchvision
from transformers import VideoMAEForVideoClassification


def build_videomae():
    # Loads the pretrained VideoMAE model for video classification
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
    # If classifier is a Linear layer, get in_features directly
    if isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
    else:
        raise ValueError("model.classifier is not a Linear layer. Please inspect the model structure.")
    model.classifier = nn.Linear(in_features, 1)
    return model
