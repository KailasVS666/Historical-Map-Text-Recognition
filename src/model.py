import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 # Using ResNet as a simple, powerful backbone

class MapTextSpotter(nn.Module):
    """
    An end-to-end model for Historical Map Text Detection, Recognition, and Linking.
    It uses a shared backbone and three specialized heads for each task.
    """
    def __init__(self, num_classes=1, num_chars=95):
        """
        Args:
            num_classes (int): Number of object classes (1: 'word').
            num_chars (int): Size of the character vocabulary (e.g., 94 printable ASCII + 1 for padding).
        """
        super().__init__()
        
        # 1. Feature Extraction Backbone (Shared)
        # We use a pre-trained ResNet to extract rich visual features from the map tile.
        resnet = resnet18(weights=None)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-2])) 
        
        # Calculate feature dimension output by the backbone
        self.feature_dim = 512 

        # 2. Detection Head (Predicts polygonal bounding boxes)
        # This head takes features and predicts the geometry (polygons).
        # Typically requires much more complex logic (e.g., FPN or Transformer Decoder)
        # For simplicity, we use a placeholder convolutional layer.
        self.detection_head = nn.Conv2d(self.feature_dim, 20, kernel_size=1) 
        
        # 3. Recognition Head (Predicts the word sequence / Transcription)
        # This head predicts the sequence of characters (the word) from the detected region's features.
        # Uses a combination of RNNs (BiLSTM) or another sequence model (Transformer Decoder).
        self.recognition_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
            # In a real model, this would feed into a sequence model (BiLSTM + CTC or Attention)
        )
        
        # 4. Linking Head (Predicts word-to-word relationships for grouping phrases)
        # This head is the unique part of your project, predicting edges in a graph.
        # Placeholder: a simple layer to predict an adjacency matrix or link scores.
        self.linking_head = nn.Sequential(
            nn.Linear(self.feature_dim * 4, 128), # Placeholder: taking features from multiple regions
            nn.ReLU(),
            nn.Linear(128, 1) # Predicts a score indicating the likelihood of a link
        )

    def forward(self, images: torch.Tensor):
        """
        Passes the input image through the shared backbone and parallel heads.
        """
        # 1. Shared Feature Extraction
        features = self.backbone(images)
        
        # 2. Detection Prediction (Placeholder output)
        detection_output = self.detection_head(features)
        
        # 3. Recognition Prediction (Placeholder output)
        recognition_features = self.recognition_head(features)
        # Actual recognition output requires complex decoding of these features per word box.
        
        # 4. Linking Prediction (Requires detected boxes/features, simplified here)
        # A real linking head would take features associated with the detected word bounding boxes.
        # We return features that would be used for linking logic outside of this forward pass.
        linking_features = features.mean(dim=[2, 3]) # Global average pool for simplicity
        
        return {
            'detection_raw': detection_output,
            'recognition_features': recognition_features,
            'linking_features': linking_features
        }

# --- Test Initialization ---
if __name__ == '__main__':
    # Create a dummy batch of 2 images (3 channels, 512x512 size)
    dummy_input = torch.randn(2, 3, 512, 512)
    
    # Initialize the model
    model = MapTextSpotter()
    
    # Run the forward pass
    output = model(dummy_input)
    
    print("\n--- Model Test Output Shapes ---")
    print(f"Detection Head Output Shape: {output['detection_raw'].shape}")
    print(f"Recognition Head Feature Shape: {output['recognition_features'].shape}")
    print(f"Linking Head Feature Shape: {output['linking_features'].shape}")