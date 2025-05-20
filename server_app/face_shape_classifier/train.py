import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from model import FACE_SHAPES, NUM_CLASSES, FaceShapeClassifier

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Paths
HIDDEN_STATES_PATH = "/media/k4/storage2/Datasets/FaceShape/features_faces_all"#faces_clear_features_aligned"#faces_hidden_states_square"
MODEL_SAVE_PATH = "models_features_faces_all"

# Create directory for saving models
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Dataset class for hidden states
class FaceShapeDataset(Dataset):
    def __init__(self, data_dir, face_shapes):
        self.tensor_paths = []
        self.labels = []
        self.face_shapes = face_shapes
        
        # Cache tensor paths for each face shape
        for i, shape in enumerate(face_shapes):
            shape_dir = os.path.join(data_dir, shape)
            print(f"Loading {shape} face data paths...")
            
            # Get all tensor files
            for root, _, files in os.walk(shape_dir):
                for file in files:
                    if file.endswith('.pt'):
                        tensor_path = os.path.join(root, file)
                        self.tensor_paths.append(tensor_path)
                        self.labels.append(i)
        
        # Convert labels to tensor for easier indexing
        self.labels = torch.tensor(self.labels)
        
        # Print dataset info
        if len(self.tensor_paths) > 0:
            print(f"Dataset loaded: {len(self.tensor_paths)} samples")
            
            # Print distribution of classes
            for i, shape in enumerate(face_shapes):
                count = (self.labels == i).sum().item()
                print(f"{shape}: {count} samples")
        else:
            raise ValueError("No data found. Check the path to hidden states.")
    
    def __len__(self):
        return len(self.tensor_paths)
    
    def __getitem__(self, idx):
        # Load tensor on-the-fly
        tensor_path = self.tensor_paths[idx]
        hidden_state = torch.load(tensor_path, map_location='cpu')
        
        # Flatten the last two dimensions
        # hidden_state.shape is typically [1, 1280, 16, 16]
        hidden_state = hidden_state.view(hidden_state.size(0), -1)[0]
        
        return hidden_state, self.labels[idx]

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    model.to(device)
    best_val_acc = 0.0
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            # Ensure outputs and labels match in format
            loss = criterion(outputs, labels.long())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'face_shape_classifier_best.pth'))
            
            # Generate classification report and confusion matrix for best model
            report = classification_report(all_labels, all_preds, target_names=FACE_SHAPES, digits=4)
            conf_matrix = confusion_matrix(all_labels, all_preds)
            
            print("\nClassification Report:")
            print(report)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=FACE_SHAPES, yticklabels=FACE_SHAPES)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_SAVE_PATH, f'confusion_matrix_epoch_{epoch+1}.png'))
            plt.close()
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'face_shape_classifier_final.pth'))
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_curves.png'))
    plt.close()
    
    return model

def main():
    # Load dataset
    full_dataset = FaceShapeDataset(HIDDEN_STATES_PATH, FACE_SHAPES)
    #my_dataset = FaceShapeDataset(HIDDEN_STATES_PATH, FACE_SHAPES)
    
    # Split dataset
    train_size = int(0.99 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
    
    # Device configuration first
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Initialize model
    # Get a sample to determine input dimension
    sample_data, _ = full_dataset[0]
    input_dim = sample_data.shape[0]  # Hidden state dimension
    hidden_dim = 378
    model = FaceShapeClassifier(input_dim, hidden_dim, NUM_CLASSES)
    model = model.to(device)  # Move model to CUDA
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss expects class indices, not one-hot vectors
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) 

    # Train the model
    num_epochs = 100
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 