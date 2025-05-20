import torch
import torch.nn as nn

# Face shapes (classes)
FACE_SHAPES = ['oval', 'oblong', 'round', 'square', 'triangle']
NUM_CLASSES = len(FACE_SHAPES)

class FaceShapeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FaceShapeClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x) 