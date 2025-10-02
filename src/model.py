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
    Final Model Structure: Shared backbone with three specialized projection heads 
    to output the exact shapes required by the loss function.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Feature Extraction Backbone (Shared)
        # We use a standard ResNet-18 structure for feature extraction
        resnet = resnet18(weights=None)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-2])) 
        self.feature_dim = 512 
        
        # --- 2. Detection Head (Outputs [B, 200, 64]) ---
        # Projects features to the fixed polygon size.
        self.det_pool = nn.AdaptiveAvgPool2d((4, 4)) # Example: Reduces spatial size to 4x4
        self.det_output_layer = nn.Linear(self.feature_dim * 4 * 4, MAX_WORDS_PER_IMAGE * MAX_VERTICES_FLAT)
        
        
        # --- 3. Recognition Head (Outputs Logits [B*6000, 99]) ---
        # Projects features to the total number of token logits.
        self.rec_pool = nn.AdaptiveAvgPool2d((1, 1)) # Global average pool
        self.rec_output_layer = nn.Linear(self.feature_dim, MAX_WORDS_PER_IMAGE * MAX_WORD_LEN * VOCAB_SIZE)
        
        
        # --- 4. Linking Head (Outputs Logits [B, 200, 200]) ---
        # Projects features to the square link matrix size.
        self.link_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.link_output_layer = nn.Linear(self.feature_dim, MAX_WORDS_PER_IMAGE * MAX_WORDS_PER_IMAGE)


    def forward(self, images: torch.Tensor):
        features = self.backbone(images)
        B = features.size(0)
        
        # 1. Detection Prediction: [B, 512*16] -> [B, 200, 64]
        det_pooled = self.det_pool(features).flatten(1)
        det_output = self.det_output_layer(det_pooled)
        detection_pred = det_output.view(B, MAX_WORDS_PER_IMAGE, MAX_VERTICES_FLAT)
        
        # 2. Recognition Prediction: [B, 512] -> [B*6000, 99]
        rec_pooled = self.rec_pool(features).flatten(1)
        rec_output = self.rec_output_layer(rec_pooled)
        recognition_logits = rec_output.view(B * MAX_WORDS_PER_IMAGE * MAX_WORD_LEN, VOCAB_SIZE)
        
        # 3. Linking Prediction: [B, 512] -> [B, 200, 200]
        link_pooled = self.link_pool(features).flatten(1)
        link_output = self.link_output_layer(link_pooled)
        linking_logits = link_output.view(B, MAX_WORDS_PER_IMAGE, MAX_WORDS_PER_IMAGE)
        
        return {
            'detection_pred': detection_pred,  
            'recognition_logits': recognition_logits, 
            'linking_logits': linking_logits
        }

if __name__ == '__main__':
    # This block verifies the output shape is correct before training
    model = MapTextSpotter()
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"Detection Pred Shape: {output['detection_pred'].shape}")
    print(f"Recognition Logits Shape: {output['recognition_logits'].shape}")
    print(f"Linking Logits Shape: {output['linking_logits'].shape}")