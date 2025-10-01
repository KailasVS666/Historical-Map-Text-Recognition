import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple

# --- Constants for File Paths and Target Shapes ---
IMAGE_ROOT = 'data/images/'
MAX_VERTICES = 32         
MAX_VERTICES_FLAT = MAX_VERTICES * 2 
MAX_WORDS_PER_IMAGE = 200 

IMAGE_SIZE = 2000         
PAD_VALUE = -1.0          # Value for padding polygon/empty word slots


# --- Vocabulary for Recognition Task ---
class Vocabulary:
    """Handles character-to-index mapping for transcription."""
    def __init__(self):
        self.char_to_index = {
            '[PAD]': 0, 
            '[UNK]': 1, 
            '[SOS]': 2, 
            '[EOS]': 3  
        }
        self.index_to_char = {v: k for k, v in self.char_to_index.items()}
        
        for i, char_code in enumerate(range(32, 127)):
            char = chr(char_code)
            if char not in self.char_to_index:
                index = len(self.char_to_index)
                self.char_to_index[char] = index
                self.index_to_char[index] = char
        
        self.pad_idx = self.char_to_index['[PAD]']
        self.vocab_size = len(self.char_to_index)

    def encode(self, text: str, max_len: int = 30) -> torch.Tensor:
        """Converts a word string into a padded tensor of token IDs."""
        tokens = [self.char_to_index.get(char, self.char_to_index['[UNK]']) for char in text]
        tokens.append(self.char_to_index['[EOS]'])
        tokens = tokens[:max_len] 
        padding_needed = max_len - len(tokens)
        if padding_needed > 0:
            tokens.extend([self.pad_idx] * padding_needed)
            
        return torch.tensor(tokens, dtype=torch.long)

MAX_WORD_LEN = 30 # Max length of the sequence of tokens


# --- Helper Function for Target Padding (Final Version) ---
def pad_targets_to_max_words(targets: List[torch.Tensor], max_words: int, feature_dim: int, pad_value: float = PAD_VALUE) -> torch.Tensor:
    """Pads a list of word tensors to the fixed size of MAX_WORDS_PER_IMAGE."""
    
    current_num_words = len(targets)
    
    if current_num_words == 0:
        return torch.full((max_words, feature_dim), pad_value, dtype=torch.float32)

    if current_num_words > max_words:
        targets = targets[:max_words]
        current_num_words = max_words

    stacked_targets = torch.stack(targets)

    if current_num_words < max_words:
        num_padding_rows = max_words - current_num_words
        padding_tensor = torch.full((num_padding_rows, feature_dim), pad_value, dtype=torch.float32)
        
        return torch.cat([stacked_targets, padding_tensor], dim=0)

    return stacked_targets


class MapTextDataset(Dataset):
    def __init__(self, annotation_file: str):
        print(f"Loading annotations from: {annotation_file}...")
        with open(annotation_file, 'r') as f:
            self.data: List[Dict[str, Any]] = json.load(f)
        
        print(f"Successfully loaded {len(self.data)} records.")
        self.vocab = Vocabulary()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        record = self.data[idx]
        image_relative_path = record['image']
        
        full_path = os.path.join(IMAGE_ROOT, image_relative_path)
        image = cv2.imread(full_path)
        
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}. Check data/images structure.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotations = self._parse_annotations(record)

        return image, annotations

    def _parse_annotations(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the raw JSON structure into Detection, Recognition, and Linking targets (Tensors).
        """
        all_polygons: List[torch.Tensor] = []
        all_token_ids: List[torch.Tensor] = [] 
        
        # New: List to hold the index of every valid word for link creation
        valid_word_indices: List[int] = [] 
        
        current_word_index = 0
        
        # 1. First Pass: Create Tensors and Map Indices for Linking
        for group in record['groups']:
            for word_data in group:
                if word_data.get('illegible') or word_data.get('truncated'):
                    continue

                # --- Detection Target (Polygon) ---
                vertices = torch.tensor(word_data['vertices'], dtype=torch.float32)
                vertices[:, 0] = vertices[:, 0] / IMAGE_SIZE
                vertices[:, 1] = vertices[:, 1] / IMAGE_SIZE
                
                num_verts = vertices.size(0)
                if num_verts > MAX_VERTICES:
                    vertices = vertices[:MAX_VERTICES]
                elif num_verts < MAX_VERTICES:
                    padding = torch.full((MAX_VERTICES - num_verts, 2), PAD_VALUE)
                    vertices = torch.cat([vertices, padding], dim=0)

                all_polygons.append(vertices.flatten()) 
                
                # --- Recognition Target (Tokens) ---
                token_ids = self.vocab.encode(word_data['text'], MAX_WORD_LEN)
                all_token_ids.append(token_ids)
                
                # Store the index of this valid word
                valid_word_indices.append(current_word_index)
                current_word_index += 1
        
        
        # 2. Linking Target: Create Adjacency Matrix (Link Graph)
        # Size: [MAX_WORDS_PER_IMAGE, MAX_WORDS_PER_IMAGE]
        link_matrix = torch.zeros((MAX_WORDS_PER_IMAGE, MAX_WORDS_PER_IMAGE), dtype=torch.float32)
        
        # Iterate over groups to define links
        for group in record['groups']:
            # Filter out truncated/illegible words from the group
            clean_group = [w for w in group if not (w.get('illegible') or w.get('truncated'))]
            
            # Find the indices of the words in the clean group
            if len(clean_group) >= 2:
                
                # This finds the absolute index of each word in the full list
                word_texts = [w['text'] for w in clean_group]
                
                # Find the index of the word object based on its order of processing (a bit fragile, 
                # but necessary since we don't have UUIDs to map the original JSON word objects).
                # We assume the order in the JSON is preserved.
                
                # Since we don't have the final ground truth index in the list, we must use the word_data objects
                
                # This is a robust way to map the raw JSON word object to its index (0 to 171)
                processed_word_objects = [
                    (all_polygons[i], all_token_ids[i]) for i in range(len(all_polygons))
                ]
                
                # We need to determine the index of each word object *as it appeared in all_polygons*
                
                # Use a simpler mapping strategy based on sequential assignment
                current_idx = 0
                for full_group in record['groups']:
                    temp_group_indices = []
                    for word in full_group:
                        if not (word.get('illegible') or word.get('truncated')):
                            temp_group_indices.append(current_idx)
                            current_idx += 1
                    
                    # Create links between sequential words in the same phrase
                    for i in range(len(temp_group_indices) - 1):
                        start_idx = temp_group_indices[i]
                        end_idx = temp_group_indices[i+1]
                        
                        # Set the forward link: word A is followed by word B
                        if start_idx < MAX_WORDS_PER_IMAGE and end_idx < MAX_WORDS_PER_IMAGE:
                            link_matrix[start_idx, end_idx] = 1.0


        # 3. Pad all targets to MAX_WORDS_PER_IMAGE size
        padded_polygons_tensor = pad_targets_to_max_words(all_polygons, MAX_WORDS_PER_IMAGE, feature_dim=MAX_VERTICES_FLAT, pad_value=PAD_VALUE)
        padded_token_ids_tensor = pad_targets_to_max_words(all_token_ids, MAX_WORDS_PER_IMAGE, feature_dim=MAX_WORD_LEN, pad_value=self.vocab.pad_idx)
        
        
        return {
            'word_polygons_padded': padded_polygons_tensor,   # Shape: [200, 64]
            'word_token_ids': padded_token_ids_tensor,       # Shape: [200, 30]
            'link_matrix_padded': link_matrix,               # Shape: [200, 200]
            'num_valid_words': current_word_index,           # Number of words before padding (172)
            'image_path': record['image']
        }


# ----------------------------------------------------------------------
#                         CORE: COLLATE FUNCTION (Unchanged)
# ----------------------------------------------------------------------

# ... (map_text_collate_fn remains the same) ...
def map_text_collate_fn(batch: List[Tuple[np.ndarray, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Custom collate function required to combine non-uniform annotation dictionaries 
    into a single batch without raising the 'equal size' RuntimeError.
    """
    images = []
    annotations_list = []

    for image, annotations in batch:
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        images.append(img_tensor)
        annotations_list.append(annotations)
    
    images_batch = torch.stack(images)
    
    return images_batch, annotations_list


# --- Test/Verification Code ---
if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    train_annotation_path = os.path.join(script_dir, '..', 'data', 'annotations', 'rumsey_train.json')
    
    test_dataset = MapTextDataset(annotation_file=train_annotation_path)
    
    # Test loading the first item
    image, annotations = test_dataset[0]
    
    print("\n--- Final Target Test Output ---")
    print(f"Loaded image shape: {image.shape} (H, W, C)")
    print(f"Total Valid Words: {annotations['num_valid_words']}")
    print(f"Padded Polygons Tensor Shape (Detection): {annotations['word_polygons_padded'].shape}") 
    print(f"Padded Token IDs Tensor Shape (Recognition): {annotations['word_token_ids'].shape}")
    print(f"Link Matrix Shape (Linking): {annotations['link_matrix_padded'].shape}")
    
    # Simple check on link matrix: should contain 1s up to the num_valid_words
    num_links = torch.sum(annotations['link_matrix_padded']).item()
    print(f"Total number of links (Edges): {int(num_links)}")