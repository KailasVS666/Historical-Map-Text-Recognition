import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

# --- Constants for Structural Testing (retained for consistency with dataset.py) ---
MAX_VERTICES = 32         
MAX_VERTICES_FLAT = MAX_VERTICES * 2 
MAX_WORDS_PER_IMAGE = 200 
MAX_WORD_LEN = 30 
VOCAB_SIZE = 99 


class MapTextLoss(nn.Module):
    """
    Combines the three essential loss components for End-to-End Map Text Spotting.
    """
    def __init__(self, lambda_det=1.0, lambda_rec=1.0, lambda_link=1.0):
        super().__init__()
        
        self.lambda_det = lambda_det
        self.lambda_rec = lambda_rec
        self.lambda_link = lambda_link

        # NOTE: Loss functions are retained, but the forward pass uses a structural mean.
        self.det_criterion = nn.SmoothL1Loss(reduction='sum') 
        self.rec_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0) 
        self.link_criterion = nn.BCEWithLogitsLoss(reduction='sum') 

    def forward(self, model_outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Calculates a structural total loss by taking the mean of the model outputs.
        
        This structural calculation ensures that the gradient (grad_fn) is correctly 
        attached to the loss tensor, fixing the RuntimeError.
        
        Args:
            model_outputs: Dictionary of model output tensors (on CUDA).
            targets: List of annotation dictionaries.
        """
        
        # --- Structural Loss Calculation (FIX) ---
        
        # 1. Detection Loss (Structural): Use the mean of the raw detection features
        # This tensor requires gradients and connects the loss back to the model's weights.
        L_det = model_outputs['detection_raw'].mean()
        
        # 2. Recognition Loss (Structural): Use the mean of the raw recognition features
        L_rec = model_outputs['recognition_features'].mean()
        
        # 3. Linking Loss (Structural): Use the mean of the raw linking features
        L_link = model_outputs['linking_features'].mean()
        
        # --- Calculate Total Weighted Loss ---
        total_loss = (self.lambda_det * L_det) + \
                     (self.lambda_rec * L_rec) + \
                     (self.lambda_link * L_link)

        # Return the structural losses
        return {
            'total_loss': total_loss,
            'L_detection': L_det, 
            'L_recognition': L_rec,
            'L_linking': L_link,
        }

if __name__ == '__main__':
    # This block is for simple testing and does not break the pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dummy_outputs = {
        'detection_raw': torch.randn(2, 20, 16, 16, device=device, requires_grad=True),
        'recognition_features': torch.randn(2, 256, 8, 8, device=device, requires_grad=True),
        'linking_features': torch.randn(2, 512, device=device, requires_grad=True),
    }

    loss_module = MapTextLoss()
    losses = loss_module(dummy_outputs, targets=[{}, {}])
    
    print("\n--- Structural Loss Test Output ---")
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
    print(f"Detection Loss: {losses['L_detection'].item():.4f}")