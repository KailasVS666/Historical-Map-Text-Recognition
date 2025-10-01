import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the custom modules you created
from dataset import MapTextDataset, map_text_collate_fn 
from model import MapTextSpotter
from loss import MapTextLoss

# --- Configuration Constants ---
NUM_EPOCHS = 2
BATCH_SIZE = 2 
LEARNING_RATE = 1e-4
TRAIN_ANNOTATION_PATH = 'data/annotations/rumsey_train.json'

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Runs a single training epoch."""
    model.train() 
    total_loss_sum = 0
    
    for i, (images, targets) in enumerate(dataloader):
        
        # 1. Prepare Data
        # Images are already batched and normalized by map_text_collate_fn
        # CORRECT: Move the image batch to the GPU
        images = images.to(device)

        # 2. Forward Pass
        optimizer.zero_grad()
        model_outputs = model(images)
        
        # 3. Calculate Loss (All tensors are now on the GPU)
        loss_dict = criterion(model_outputs, targets=targets)
        total_loss = loss_dict['total_loss']
        
        # 4. Backward Pass and Optimization
        total_loss.backward()
        optimizer.step()
        
        total_loss_sum += total_loss.item()
        
        if i % 10 == 0:
             print(f"  Batch {i}/{len(dataloader)} | Total Loss: {total_loss.item():.4f} | Det: {loss_dict['L_detection'].item():.4f}")

    return total_loss_sum / len(dataloader)

def main():
    # This line correctly detects your GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    train_dataset = MapTextDataset(annotation_file=TRAIN_ANNOTATION_PATH)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=map_text_collate_fn 
    )

    # --- 2. Initialize Model, Loss, and Optimizer ---
    # The .to(device) call moves the model weights to the GPU
    model = MapTextSpotter().to(device)
    criterion = MapTextLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Structural Training Loop...")
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} Finished ---")
        print(f"Average Training Loss: {avg_loss:.4f}")
        
    print("\n✅ End-to-End Pipeline Structural Check Complete!")


if __name__ == '__main__':
    main()