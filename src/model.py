import torch
import torch.nn as nn
import torchvision
from transformers import VideoMAEForVideoClassification


def build_videomae():
    # Loads the pretrained VideoMAE model for video classification
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
    # If classifier (final layer) is a Linear layer, get in_features directly
    if isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
    else:
        raise ValueError("model.classifier is not a Linear layer. Please inspect the model structure.")
    model.classifier = nn.Linear(in_features, 1) #maps all those features to a single output value
    return model


def attention_to_frame_weights(attention_map, num_frames=16, patches_per_frame=196):
    """
    Convert attention map to frame weights for explainability
    
    Input: attention_map shape [batch_size, num_heads, num_tokens, num_tokens]
    Example: [1, 12, 3136, 3136] - 12 attention heads, 3136 tokens
    
    Output: frame_weights shape [num_frames] 
    Example: [0.1, 0.2, 0.3, ..., 0.9] - 16 numbers, one per frame
    """
    
    # Step 1: Average across attention heads
    # Why? Each head focuses on different patterns, we want the average
    # Input: [1, 12, 3136, 3136] 
    # Output: [1, 3136, 3136] #this gets 1 number per row
    attention_avg = attention_map.mean(dim=1) #average all attention heads on 1 frame
    
    # Step 2: Get how much attention all heads get
    # Why? We want to know "how important is each patch?"
    # Input: [1, 3136, 3136]
    # Output: [1, 3136] - one number per head
    token_attention = attention_avg.mean(dim=1) #average each row of the 3136x3136 row
    
    # Step 3: Group patches by frames and average
    frame_weights = []
    for frame_idx in range(num_frames):
        # Calculate which patches belong to this frame
        # Frame 0: patches 0-195, Frame 1: patches 196-391, etc.
        start_patch = frame_idx * patches_per_frame      # Frame 0: 0, Frame 1: 196
        end_patch = (frame_idx + 1) * patches_per_frame # Frame 0: 196, Frame 1: 392
        
        # Get attention weights for current frame's patches
        # Example: frame_patches = [0.1, 0.2, 0.3, ..., 0.1] (196 numbers)

        # frame_0: patches 0 - 196 = 196 attention weights
        # frame_1: patches 196 - 391 = 196 attention weights
        frame_patches = token_attention[0, start_patch:end_patch]
        
        # Average the attention weights for this frame
        # Example: frame_weight = 0.15 (average of 196 patch weights)
        frame_weight = frame_patches.mean().item() #converts from tensor num to python num
        frame_weights.append(frame_weight)
    
    return frame_weights


# def test_attention_conversion():
#     """Test the attention_to_frame_weights function"""
#     print("Testing attention conversion...")
    
#     # Create a dummy attention map
#     batch_size, num_heads, num_tokens = 1, 12, 3136
#     dummy_attention = torch.randn(batch_size, num_heads, num_tokens, num_tokens)
    
#     # Convert to frame weights
#     frame_weights = attention_to_frame_weights(dummy_attention)

#     # Find the frame with highest attention
#     max_weight = max(frame_weights)
#     max_frame_idx = frame_weights.index(max_weight)
#     print(f"Frame {max_frame_idx} has highest attention: {max_weight}")
    
#     print(f"Frame weights: {frame_weights}")
#     print(f"Number of frames: {len(frame_weights)}")
#     print(f"Frame weights sum: {sum(frame_weights):.3f}")
    
#     return frame_weights


if __name__ == "__main__":
    print("Testing attention capture...")
    # frame_weights = test_attention_conversion()
    print("Test completed!")