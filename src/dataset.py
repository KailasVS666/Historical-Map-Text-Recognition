import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple

# --- Constants for File Paths (Relative to Project Root) ---
IMAGE_ROOT = 'data/images/'

class MapTextDataset(Dataset):
    """
    A PyTorch Dataset class for the Historical Map Text Spotting task.
    Loads images and parses polygonal annotations.
    """
    def __init__(self, annotation_file: str):
        print(f"Loading annotations from: {annotation_file}...")
        with open(annotation_file, 'r') as f:
            self.data: List[Dict[str, Any]] = json.load(f)
        
        print(f"Successfully loaded {len(self.data)} records.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads and returns a single data sample. Returns raw NumPy image and dictionary.
        """
        record = self.data[idx]
        image_relative_path = record['image']
        
        # 1. Image Loading and Path Construction
        full_path = os.path.join(IMAGE_ROOT, image_relative_path)
        
        # Load image in BGR format
        image = cv2.imread(full_path)
        
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}. Check data/images structure.")

        # Convert image to RGB for consistency
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Extract and Prepare Annotations
        annotations = self._parse_annotations(record)

        # Return NumPy array and Python dictionary (will be handled by collate_fn)
        return image, annotations

    def _parse_annotations(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Processes the raw JSON structure into a simplified list of targets."""
        all_polygons = []
        all_transcriptions = []
        all_word_data = [] 

        for group in record['groups']:
            for word_data in group:
                if word_data.get('illegible') or word_data.get('truncated'):
                    continue

                all_polygons.append(word_data['vertices'])
                all_transcriptions.append(word_data['text'])
                all_word_data.append(word_data)
        
        return {
            'word_polygons': all_polygons,
            'word_transcriptions': all_transcriptions,
            'word_objects': all_word_data, 
            'image_path': record['image']
        }


# ----------------------------------------------------------------------
#                         CORE FIX: COLLATE FUNCTION
# ----------------------------------------------------------------------

def map_text_collate_fn(batch: List[Tuple[np.ndarray, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Custom collate function required to combine non-uniform annotation dictionaries 
    into a single batch without raising the 'equal size' RuntimeError.
    """
    images = []
    annotations_list = []

    for image, annotations in batch:
        # 1. Convert image to PyTorch Tensor: HWC -> CWH, Normalize to 0-1
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        images.append(img_tensor)
        
        # 2. Keep annotations as a simple list of dictionaries
        annotations_list.append(annotations)
    
    # Stack the images into a single Tensor batch
    images_batch = torch.stack(images)
    
    # Return the batched images and the list of annotation dictionaries
    return images_batch, annotations_list


# --- Test/Verification Code (Optional, but Recommended to run in a test script) ---
if __name__ == '__main__':
    train_annotation_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'annotations', 'rumsey_train.json')
    test_dataset = MapTextDataset(annotation_file=train_annotation_path)
    
    image, annotations = test_dataset[0]
    
    print("\n--- Test Output ---")
    print(f"Loaded image shape: {image.shape} (H, W, C)")
    print(f"Total polygons extracted: {len(annotations['word_polygons'])}")
    print(f"First 3 transcriptions: {annotations['word_transcriptions'][:3]}")