import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

# --- Constants for Loss Calculation (derived from model/dataset) ---
MAX_VERTICES_FLAT = 64
MAX_WORDS_PER_IMAGE = 200
MAX_WORD_LEN = 30 
VOCAB_SIZE = 99 


class MapTextLoss(nn.Module):
    """
    Final loss module combining Detection (SmoothL1), Recognition (CE), and Linking (BCE).
    """
    def __init__(self, lambda_det=10.0, lambda_rec=1.0, lambda_link=5.0):
        super().__init__()
        
        # Set realistic weights for the tasks
        self.lambda_det = lambda_det 
        self.lambda_rec = lambda_rec 
        self.lambda_link = lambda_link

        # 1. Detection: Smooth L1 Loss for polygon regression (more robust than L2)
        self.det_criterion = nn.SmoothL1Loss(reduction='none') 
        
        # 2. Recognition: Cross-Entropy Loss (for token sequence prediction)
        self.rec_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0) 
        
        # 3. Linking: BCE with Logits Loss (for binary link prediction)
        self.link_criterion = nn.BCEWithLogitsLoss(reduction='none') 

    def forward(self, model_outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Calculates the weighted total loss using all finalized targets and predictions.
        """
        device = model_outputs['detection_pred'].device
        total_loss_sum = 0
        
        # The batch size (B) is the number of samples in the targets list
        B = len(targets)
        
        # Split model predictions by sample index
        det_preds = model_outputs['detection_pred'] # [B, 200, 64]
        rec_logits = model_outputs['recognition_logits'] # [B*6000, 99]
        link_logits = model_outputs['linking_logits'] # [B, 200, 200]
        
        
        # --- Batch Processing and Loss Accumulation ---
        
        L_det_accum = 0
        L_rec_accum = 0
        L_link_accum = 0

        for i in range(B):
            
            # --- 1. Detection Loss (Polygons) ---
            target_polygons = targets[i]['word_polygons_padded'].to(device) # [200, 64]
            pred_polygons = det_preds[i] # [200, 64]

            # Mask to ignore padding slots (-1.0)
            valid_mask = (target_polygons != -1.0) 
            
            # Calculate L1 loss only on valid, non-padded polygon coordinates
            L_det_sample = self.det_criterion(pred_polygons, target_polygons)
            L_det_sample = L_det_sample[valid_mask].mean() # Mean over valid elements
            L_det_accum += L_det_sample
            
            
            # --- 2. Recognition Loss (Tokens) ---
            target_tokens = targets[i]['word_token_ids'].long().to(device) # [200, 30]
            
            # Isolate the segment of the flattened logits corresponding to this sample
            start_idx = i * (MAX_WORDS_PER_IMAGE * MAX_WORD_LEN)
            end_idx = start_idx + (MAX_WORDS_PER_IMAGE * MAX_WORD_LEN)
            pred_tokens_logits = rec_logits[start_idx:end_idx] # [6000, 99]
            
            # Calculate Cross-Entropy Loss on the flattened tokens
            L_rec_sample = self.rec_criterion(pred_tokens_logits, target_tokens.view(-1))
            L_rec_sample = L_rec_sample.mean() # Mean over all tokens (valid + PAD)
            L_rec_accum += L_rec_sample
            
            
            # --- 3. Linking Loss (Adjacency Matrix) ---
            target_links = targets[i]['link_matrix_padded'].to(device) # [200, 200]
            pred_links = link_logits[i] # [200, 200]
            
            # We only evaluate links among the actually processed words (up to num_valid_words)
            num_valid = targets[i]['num_valid_words']
            
            valid_area_pred = pred_links[:num_valid, :num_valid]
            valid_area_target = target_links[:num_valid, :num_valid]
            
            L_link_sample = self.link_criterion(valid_area_pred, valid_area_target)
            L_link_sample = L_link_sample.mean() # Mean over valid link area
            L_link_accum += L_link_sample

        
        # --- Final Total Loss Calculation ---
        # Average the accumulated losses across the batch size (B)
        L_det_final = L_det_accum / B
        L_rec_final = L_rec_accum / B
        L_link_final = L_link_accum / B
        
        total_loss = (self.lambda_det * L_det_final) + \
                     (self.lambda_rec * L_rec_final) + \
                     (self.lambda_link * L_link_final)

        # Return the losses
        return {
            'total_loss': total_loss,
            'L_detection': L_det_final, 
            'L_recognition': L_rec_final,
            'L_linking': L_link_final,
        }