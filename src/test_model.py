#!/usr/bin/env python3
"""
🎯 Collision Forecast Model Test Script
=====================================

This script helps you understand how your VideoMAE model predicts crashes.
Think of it like a "model doctor" that checks if your AI is working correctly!

What this script does:
1. Loads your trained model
2. Tests it on sample videos
3. Shows you what the model "sees" and why it makes predictions
4. Creates visualizations to help you understand the results

Author: Your AI Teacher
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_videomae, attention_to_frame_weights
from dataloader import CollisionForecastDataset
from torch.utils.data import DataLoader

# Set up pretty plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelTester:
    """
    🧪 Model Tester Class
    
    This class is like a "model doctor" that:
    - Loads your trained model
    - Tests it on different scenarios
    - Explains what the model is thinking
    - Shows you visual results
    """
    
    def __init__(self, model_path=None, device='auto'):
        """
        Initialize the model tester
        
        Args:
            model_path: Path to your trained model (.pth file)
            device: 'cuda', 'cpu', or 'auto' (let's the script decide)
        """
        print("🔧 Setting up Model Tester...")
        
        # Set up device (GPU if available, otherwise CPU)
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"📱 Using device: {self.device}")
        
        # Build the model (same as in training)
        print("🏗️ Building VideoMAE model...")
        self.model = build_videomae()
        
        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            print(f"📥 Loading trained model from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✅ Model loaded successfully!")
        else:
            print("⚠️ No trained model found. Using untrained model for testing.")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print("🎉 Model Tester ready!")
    
    def test_single_clip(self, clip_path, expected_label=None):
        """
        Test the model on a single video clip
        
        Args:
            clip_path: Path to the .npy file (16-frame video clip)
            expected_label: What you expect the result to be (for comparison)
        
        Returns:
            Dictionary with prediction results and explanations
        """
        print(f"\n🎬 Testing clip: {clip_path}")
        
        # Load the video clip
        try:
            clip = np.load(clip_path)
            print(f"📊 Clip shape: {clip.shape}")
        except Exception as e:
            print(f"❌ Error loading clip: {e}")
            return None
        
        # Normalize the clip (same as in dataloader)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
        
        clip = clip / 255.0  # Scale from [0,255] to [0,1]
        clip = (clip - mean) / std  # Normalize channel-wise
        
        # Convert to tensor and add batch dimension
        clip_tensor = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        clip_tensor = clip_tensor.to(self.device)
        
        # Reorder dimensions for VideoMAE (batch, channels, frames, height, width)
        clip_tensor = clip_tensor.permute(0, 2, 1, 3, 4)
        
        print(f"🔧 Input tensor shape: {clip_tensor.shape}")
        
        # Make prediction
        with torch.no_grad():  # Don't compute gradients during testing
            prediction = self.model(clip_tensor)
            
        # Extract the prediction value
        predicted_time = prediction.logits.item()
        
        # Interpret the prediction
        interpretation = self._interpret_prediction(predicted_time)
        
        # Create results dictionary
        results = {
            'clip_path': clip_path,
            'predicted_time': predicted_time,
            'interpretation': interpretation,
            'expected_label': expected_label,
            'clip_shape': clip.shape,
            'model_input_shape': clip_tensor.shape
        }
        
        # Print results in a nice format
        self._print_results(results)
        
        return results
    
    def _interpret_prediction(self, predicted_time):
        """
        Convert the model's numerical output into human-readable interpretation
        
        Args:
            predicted_time: The model's raw prediction (seconds)
        
        Returns:
            Dictionary with interpretation details
        """
        # Your model predicts "seconds until crash"
        # Lower values = crash happening soon
        # Higher values = safer driving
        
        if predicted_time <= 0:
            risk_level = "🚨 IMMINENT CRASH"
            explanation = "Model predicts crash is happening right now!"
        elif predicted_time <= 1:
            risk_level = "⚠️ HIGH RISK"
            explanation = "Model predicts crash within 1 second"
        elif predicted_time <= 3:
            risk_level = "🟡 MEDIUM RISK"
            explanation = "Model predicts crash within 3 seconds"
        elif predicted_time <= 5:
            risk_level = "🟢 LOW RISK"
            explanation = "Model predicts crash within 5 seconds"
        else:
            risk_level = "✅ SAFE"
            explanation = "Model predicts safe driving (no crash soon)"
        
        return {
            'risk_level': risk_level,
            'explanation': explanation,
            'seconds_until_crash': predicted_time
        }
    
    def _print_results(self, results):
        """
        Print test results in a nice, readable format
        """
        print("\n" + "="*60)
        print("🎯 MODEL PREDICTION RESULTS")
        print("="*60)
        
        print(f"📁 Clip: {os.path.basename(results['clip_path'])}")
        print(f"🔢 Raw Prediction: {results['predicted_time']:.3f} seconds")
        print(f"⚠️ Risk Level: {results['interpretation']['risk_level']}")
        print(f"💡 Explanation: {results['interpretation']['explanation']}")
        
        if results['expected_label']:
            print(f"🎯 Expected: {results['expected_label']}")
            # You could add comparison logic here
        
        print("="*60)
    
    def test_multiple_clips(self, test_dataset, num_samples=5):
        """
        Test the model on multiple clips from your dataset
        
        Args:
            test_dataset: Your CollisionForecastDataset
            num_samples: How many clips to test
        """
        print(f"\n🧪 Testing {num_samples} random clips from dataset...")
        
        # Create a data loader
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        
        results = []
        
        for i, (clip, target) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            # Get the clip path (this is a bit hacky, but works for testing)
            clip_path = f"Sample {i+1}"
            
            # Make prediction
            clip = clip.to(self.device)
            clip = clip.permute(0, 2, 1, 3, 4)  # Reorder for VideoMAE
            
            with torch.no_grad():
                prediction = self.model(clip)
                predicted_time = prediction.logits.item()
                actual_time = target.item()
            
            # Interpret results
            interpretation = self._interpret_prediction(predicted_time)
            
            result = {
                'sample_id': i + 1,
                'predicted_time': predicted_time,
                'actual_time': actual_time,
                'interpretation': interpretation,
                'error': abs(predicted_time - actual_time)
            }
            
            results.append(result)
            
            # Print individual result
            print(f"\n📊 Sample {i+1}:")
            print(f"   Predicted: {predicted_time:.3f}s")
            print(f"   Actual: {actual_time:.3f}s")
            print(f"   Error: {result['error']:.3f}s")
            print(f"   Risk: {interpretation['risk_level']}")
        
        # Calculate overall statistics
        self._print_test_statistics(results)
        
        return results
    
    def _print_test_statistics(self, results):
        """
        Print overall test statistics
        """
        print("\n" + "="*60)
        print("📈 OVERALL TEST STATISTICS")
        print("="*60)
        
        errors = [r['error'] for r in results]
        predicted_times = [r['predicted_time'] for r in results]
        actual_times = [r['actual_time'] for r in results]
        
        print(f"📊 Number of samples tested: {len(results)}")
        print(f"📊 Average prediction error: {np.mean(errors):.3f} seconds")
        print(f"📊 Standard deviation of error: {np.std(errors):.3f} seconds")
        print(f"📊 Min error: {min(errors):.3f} seconds")
        print(f"📊 Max error: {max(errors):.3f} seconds")
        
        print(f"\n🎯 Prediction Range: {min(predicted_times):.3f}s to {max(predicted_times):.3f}s")
        print(f"🎯 Actual Range: {min(actual_times):.3f}s to {max(actual_times):.3f}s")
        
        # Count risk levels
        risk_counts = {}
        for result in results:
            risk = result['interpretation']['risk_level']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        print(f"\n⚠️ Risk Level Distribution:")
        for risk, count in risk_counts.items():
            percentage = (count / len(results)) * 100
            print(f"   {risk}: {count} samples ({percentage:.1f}%)")
        
        print("="*60)
    
    def visualize_predictions(self, results, save_path=None):
        """
        Create visualizations of the test results
        
        Args:
            results: List of test results from test_multiple_clips
            save_path: Where to save the plot (optional)
        """
        print("\n📊 Creating visualizations...")
        
        # Extract data
        predicted_times = [r['predicted_time'] for r in results]
        actual_times = [r['actual_time'] for r in results]
        errors = [r['error'] for r in results]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🎯 Collision Forecast Model Test Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Predicted vs Actual
        ax1.scatter(actual_times, predicted_times, alpha=0.7, s=100)
        ax1.plot([0, max(actual_times)], [0, max(actual_times)], 'r--', alpha=0.8, label='Perfect Prediction')
        ax1.set_xlabel('Actual Time (seconds)')
        ax1.set_ylabel('Predicted Time (seconds)')
        ax1.set_title('🎯 Predicted vs Actual Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error Distribution
        ax2.hist(errors, bins=10, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Prediction Error (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('📊 Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Prediction Distribution
        ax3.hist(predicted_times, bins=10, alpha=0.7, color='green', edgecolor='black', label='Predicted')
        ax3.hist(actual_times, bins=10, alpha=0.7, color='blue', edgecolor='black', label='Actual')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('📈 Time Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error vs Predicted
        ax4.scatter(predicted_times, errors, alpha=0.7, s=100, color='red')
        ax4.set_xlabel('Predicted Time (seconds)')
        ax4.set_ylabel('Error (seconds)')
        ax4.set_title('❌ Error vs Prediction')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {save_path}")
        
        plt.show()
        
        print("✅ Visualizations created!")


def main():
    """
    🚀 Main function to run the model tests
    """
    print("🎯 Welcome to the Collision Forecast Model Tester!")
    print("="*60)
    
    # Check if we have a trained model
    checkpoint_dir = "checkpoints"
    model_path = None
    
    if os.path.exists(checkpoint_dir):
        model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if model_files:
            model_path = os.path.join(checkpoint_dir, model_files[0])
            print(f"📁 Found trained model: {model_path}")
        else:
            print("⚠️ No trained model found in checkpoints directory")
    else:
        print("⚠️ No checkpoints directory found")
    
    # Initialize the tester
    tester = ModelTester(model_path=model_path)
    
    # Test on dataset if available
    try:
        # Try to load test dataset
        test_dataset = CollisionForecastDataset(
            csv_path="../labels/test_clip_labels.csv",
            split="test"
        )
        print(f"📊 Loaded test dataset with {len(test_dataset)} samples")
        
        # Test multiple clips
        results = tester.test_multiple_clips(test_dataset, num_samples=10)
        
        # Create visualizations
        tester.visualize_predictions(results, save_path="test_results.png")
        
    except Exception as e:
        print(f"⚠️ Could not load test dataset: {e}")
        print("💡 Make sure you have test_clip_labels.csv in the labels directory")
    
    print("\n🎉 Model testing completed!")
    print("💡 Check the results above to understand how your model is performing.")


if __name__ == "__main__":
    main() 