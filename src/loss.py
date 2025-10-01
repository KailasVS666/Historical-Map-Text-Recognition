import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class MapTextLoss(nn.Module):
    """
    Combines the three essential loss components for End-to-End Map Text Spotting.
    """
    def __init__(self, lambda_det=1.0, lambda_rec=1.0, lambda_link=1.0):
        super().__init__()
        
        # Loss weighting coefficients
        self.lambda_det = lambda_det
        self.lambda_rec = lambda_rec
        self.lambda_link = lambda_link

        # Loss Function Placeholders
        self.det_criterion = nn.SmoothL1Loss(reduction='mean')
        self.rec_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.link_criterion = nn.BCEWithLogitsLoss(reduction='mean') 

    def forward(self, model_outputs: Dict[str, torch.Tensor], targets: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Calculates the weighted total loss.
        """
        
        # FIX: Retrieve the device (cuda) from the model's output tensor
        device = model_outputs['detection_raw'].device
        
        # 1. Detection Loss (Dummy targets created on the correct device)
        dummy_det_pred = model_outputs['detection_raw'].flatten(1)
        # FIX: Add device=device
        dummy_det_target = torch.randn_like(dummy_det_pred, device=device) 
        L_det = self.det_criterion(dummy_det_pred, dummy_det_target)
        
        # 2. Recognition Loss (Dummy targets created on the correct device)
        dummy_rec_pred = model_outputs['recognition_features'].mean(dim=[2, 3])
        # FIX: Add device=device
        dummy_rec_target = torch.randint(
            0, dummy_rec_pred.shape[1], (dummy_rec_pred.shape[0],), device=device
        )
        L_rec = self.rec_criterion(dummy_rec_pred, dummy_rec_target)
        
        # 3. Linking Loss (Dummy targets created on the correct device)
        dummy_link_pred = model_outputs['linking_features'].mean(dim=1)
        # FIX: Add device=device
        dummy_link_target = torch.randint(
            0, 2, dummy_link_pred.shape, device=device
        )
        L_link = self.link_criterion(dummy_link_pred, dummy_link_target.float())

        # --- Calculate Total Weighted Loss ---
        total_loss = (self.lambda_det * L_det) + \
                     (self.lambda_rec * L_rec) + \
                     (self.lambda_link * L_link)

        return {
            'total_loss': total_loss,
            'L_detection': L_det,
            'L_recognition': L_rec,
            'L_linking': L_link,
        }

if __name__ == '__main__':
    # Test with Cuda device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dummy_outputs = {
        'detection_raw': torch.randn(2, 20, 16, 16, device=device),
        'recognition_features': torch.randn(2, 256, 8, 8, device=device),
        'linking_features': torch.randn(2, 512, device=device),
    }

    loss_module = MapTextLoss()
    losses = loss_module(dummy_outputs, targets={})
    
    print("\n--- Loss Function Test Output ---")
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
    print(f"Detection Loss: {losses['L_detection'].item():.4f}")
    print(f"Recognition Loss: {losses['L_recognition'].item():.4f}")
    print(f"Linking Loss: {losses['L_linking'].item():.4f}")