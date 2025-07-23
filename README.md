# Collision Forecast
A VideoMAE-based model for early crash risk prediction from dashcam footage.

# Project Overview
Collision Forecast uses a fine-tuned VideoMAE transformer to forecast the risk of collisions from dashcam video. Instead of simply detecting crashes, this model predicts how likely a crash is about to occur by analyzing the motion and scene context before impact.

This project simulates early warning systems in ADAS (Advanced Driver-Assistance Systems) and autonomous vehicles, offering real-time crash likelihood scores several seconds before a collision occurs.

# What Makes It Different?
- Forecasts crash risk (regression), not just detection
- Uses VideoMAE, a powerful transformer-based video model
- Learns motion cues from 16-frame sequences before crashes
- Built entirely in PyTorch + OpenCV, fully containerized
- Deployed on Google Cloud Run with Docker + CI/CD

# Dataset
The model is trained on the Nexar AI Dashcam Challenge dataset.
Raw dashcam footage containing annotated crash and non-crash events
16-frame sequence leading up to the event timestamp were extracted from each training sample
Frames are extracted using OpenCV and stacked into tensors for model input

# Model
Architecture: VideoMAE
Video Masked Autoencoder for Self-supervised Video Pretraining (Fang et al., 2022) (https://doi.org/10.48550/arXiv.2203.12602)
Pretrained transformer that models spatiotemporal patterns in video
Fine-tuned to perform regression on crash likelihood
Input shape: (batch_size, channels=3, frames=16, height, width)

# Run with Docker
``` bash
git clone https://github.com/KosiAtupulazi/collision-forecast.git
cd collision-forecast

# Build and run
docker build -t forecast-app .
docker run -p 8501:8501 forecast-app
```
