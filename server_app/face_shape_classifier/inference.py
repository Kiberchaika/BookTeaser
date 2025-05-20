import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
import open_clip

# Import the classifier model definition from the training script
from face_shape_classifier.model import FaceShapeClassifier, FACE_SHAPES

# MODEL_PATH from current directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pth')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global model variables
clip_model = None
clip_preprocessor = None
classifier_model = None

def initialize_models():
    """Initialize the models globally"""
    global clip_model, classifier_model, clip_preprocessor
    
    if clip_model is None or classifier_model is None:
        # Determine input dimension from CLIP model
        model, _, clip_preprocessor = open_clip.create_model_and_transforms('ViT-H-14', pretrained='metaclip_altogether')
        model = model.to(device)
        model.eval()
        
        # Get hidden state dimensions
        model_dtype = next(model.parameters()).dtype
        print(f"Model is using {model_dtype} precision")
        
        input_dim = 1024
        hidden_dim = 378  # Using the same HIDDEN_DIM as before
        num_classes = len(FACE_SHAPES)
        
        # Initialize the classifier with the same architecture
        classifier = FaceShapeClassifier(input_dim, hidden_dim, num_classes)
        classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        classifier.to(device)
        classifier.eval()
        
        clip_model = model
        classifier_model = classifier

def make_square_with_white_background(image, target_size=244):
    """
    Makes the image square with a white background and resizes it
    while preserving aspect ratio with a maximum size of target_size
    """
    # Get original dimensions
    width, height = image.size
    
    # Calculate the new size while preserving aspect ratio
    if width > height:
        new_width = min(width, target_size)
        new_height = int(height * (new_width / width))
    else:
        new_height = min(height, target_size)
        new_width = int(width * (new_height / height))
    
    # Resize the image preserving aspect ratio
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a square white background
    square_size = max(new_width, new_height)
    square_size = min(square_size, target_size)  # Ensure max size is target_size
    
    square_image = Image.new('RGB', (square_size, square_size), (255, 255, 255))
    
    # Calculate position to paste (center)
    paste_x = (square_size - new_width) // 2
    paste_y = (square_size - new_height) // 2
    
    # Paste the resized image onto the white square
    square_image.paste(resized_image, (paste_x, paste_y))
    
    return square_image

def extract_clip_features(image_path):
    global clip_model, clip_preprocessor
    
    """Extract CLIP features from an image using standard forward pass"""
    # Load and clip_preprocessor the image
    
    image = Image.open(image_path).convert('RGB')

    # Make image square with white background and resize
    image = make_square_with_white_background(image, target_size=244)
    
    image_input = clip_preprocessor(image).unsqueeze(0).to(device)
    
    # Match the model precision
    model_dtype = next(clip_model.parameters()).dtype
    image_input = image_input.to(dtype=model_dtype)
    
    # Extract features using standard forward pass
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        
        # Flatten the features to match the classifier's expected input
        hidden_state = image_features.view(image_features.size(0), -1)
        
    return hidden_state

def preprocess_image(image_path):
    """
    Process an image and return the predicted face shape class
    Returns:
        str: The predicted face shape class
    """
    global classifier_model, clip_preprocessor

    # Extract features
    features = extract_clip_features(image_path)
    
    if features is None:
        return None
    
    # Predict face shape
    with torch.no_grad():
        outputs = classifier_model(features)
        _, predicted = torch.max(outputs, 1)
        predicted_class_idx = predicted.item()
        predicted_class = FACE_SHAPES[predicted_class_idx]
    
    return predicted_class

def unload_model():
    """Unload models and free up memory"""
    global clip_model, classifier_model
    
    if clip_model is not None:
        # Move models to CPU first
        clip_model.cpu()
        classifier_model.cpu()
        
        # Delete models
        del clip_model
        del classifier_model
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        clip_model = None
        classifier_model = None


if __name__ == "__main__":
    # Example usage
    # path to test.jpg in the same directory as the script
    test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.jpg')

    # Initialize models when the module is imported
    initialize_models()

    if os.path.exists(test_image_path):
        result = preprocess_image(test_image_path)
        print(f"Predicted face shape: {result}") 

    unload_model()