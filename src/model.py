import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 

# Constants derived from src/dataset.py targets
MAX_VERTICES_FLAT = 64
MAX_WORDS_PER_IMAGE = 200
MAX_WORD_LEN = 30
VOCAB_SIZE = 99


class MapTextSpotter(nn.Module):
    """
    An end-to-end model for Historical Map Text Detection, Recognition, and Linking.
    It uses a shared backbone and three specialized heads for each task.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Feature Extraction Backbone (Shared)
        # We use a pre-trained ResNet-18 (without weights for structural test)
        resnet = resnet18(weights=None)
        # Take layers up to layer4, outputting spatial features
        self.backbone = nn.Sequential(*(list(resnet.children())[:-2])) 
        
        # Output feature map from ResNet-18 is 512 channels
        self.feature_dim = 512 
        
        # --- 2. Detection Head (Predicts Polygons [200, 64]) ---
        # The detection prediction must be a fixed size, so we project the features.
        
        # Simple projection: Reduce features spatially, then use Linear layer to flatten to final size.
        # Downscale feature map (e.g., from [512, 16, 16] to [512, 4, 4] -> 8192 features)
        self.det_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Final Detection Layer: Maps pooled features to the flattened polygon size (200 * 64)
        self.det_output_layer = nn.Linear(self.feature_dim * 4 * 4, MAX_WORDS_PER_IMAGE * MAX_VERTICES_FLAT)
        
        
        # --- 3. Recognition Head (Predicts Tokens [200 * 30, 99]) ---
        # This head needs to output logits for the vocabulary size (99) for every token position (200 * 30).
        
        # Placeholder: Pool features and project to token logits.
        self.rec_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Maps pooled features (512) to the total number of prediction slots (200 words * 30 tokens * 99 classes)
        self.rec_output_layer = nn.Linear(self.feature_dim, MAX_WORDS_PER_IMAGE * MAX_WORD_LEN * VOCAB_SIZE)
        
        
        # --- 4. Linking Head (Predicts Links [200, 200]) ---
        # This head needs to predict an adjacency matrix.
        
        # Simple projection: Global Average Pool features and project to a square matrix size.
        self.link_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Maps pooled features (512) to the squared link matrix size (200 * 200)
        self.link_output_layer = nn.Linear(self.feature_dim, MAX_WORDS_PER_IMAGE * MAX_WORDS_PER_IMAGE)


    def forward(self, images: torch.Tensor):
        """
        Passes the input image through the shared backbone and parallel heads.
        """
        # 1. Shared Feature Extraction
        features = self.backbone(images) # [B, 512, H/32, W/32]
        B = features.size(0)
        
        # --- 2. Detection Prediction ---
        # [B, 512, 16, 16] -> [B, 512*4*4] -> [B, 200 * 64]
        det_pooled = self.det_pool(features).flatten(1)
        det_output = self.det_output_layer(det_pooled)
        # Reshape output to match target structure: [B, MAX_WORDS_PER_IMAGE, MAX_VERTICES_FLAT]
        det_output = det_output.view(B, MAX_WORDS_PER_IMAGE, MAX_VERTICES_FLAT)
        
        # --- 3. Recognition Prediction ---
        # [B, 512, 16, 16] -> [B, 512] -> [B, 200 * 30 * 99]
        rec_pooled = self.rec_pool(features).flatten(1)
        rec_output = self.rec_output_layer(rec_pooled)
        # Reshape output to match loss input: [B * MAX_WORDS * MAX_WORD_LEN, VOCAB_SIZE]
        rec_output = rec_output.view(B * MAX_WORDS_PER_IMAGE * MAX_WORD_LEN, VOCAB_SIZE)
        
        # --- 4. Linking Prediction ---
        # [B, 512, 16, 16] -> [B, 512] -> [B, 200 * 200]
        link_pooled = self.link_pool(features).flatten(1)
        link_output = self.link_output_layer(link_pooled)
        # Reshape output to match target structure: [B, MAX_WORDS_PER_IMAGE, MAX_WORDS_PER_IMAGE]
        link_output = link_output.view(B, MAX_WORDS_PER_IMAGE, MAX_WORDS_PER_IMAGE)
        
        return {
            'detection_pred': det_output,  # [B, 200, 64]
            'recognition_logits': rec_output, # [B*6000, 99]
            'linking_logits': link_output # [B, 200, 200]
        }

# --- Test Initialization ---
if __name__ == '__main__':
    # Create a dummy batch of 2 images (3 channels, e.g., 512x512 size)
    dummy_input = torch.randn(2, 3, 512, 512)
    
    # Initialize the model
    model = MapTextSpotter()
    
    # Run the forward pass
    output = model(dummy_input)
    
    print("\n--- Final Model Output Shapes ---")
    print(f"Detection Pred Shape: {output['detection_pred'].shape}")
    print(f"Recognition Logits Shape: {output['recognition_logits'].shape}")
    print(f"Linking Logits Shape: {output['linking_logits'].shape}")