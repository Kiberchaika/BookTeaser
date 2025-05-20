import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from PIL import Image
import sys
import numpy as np
from torchvision import transforms
import open_clip
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from keras.utils import get_file
import bz2
import tempfile

# Import the classifier model definition from the training script
from train_face_shape_classifier import FaceShapeClassifier, FACE_SHAPES

# Set paths
MODEL_PATH = "/home/k4/Projects/BookFace_2025/models_features_faces_all/face_shape_classifier_best.pth"
TEST_IMAGE_PATH = "/home/k4/Projects/BookFace_2025/shape_faces_test/Мужчина/3/1.JPG"
TEST_IMAGE_PATH = "/home/k4/Projects/BookFace_2025/shape_faces_test/Женщина/x2.JPG"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Landmarks model URL
LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def align_face(image_path, output_size=512, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
    """Aligns a face in an image using landmarks detection"""
    # Download and unpack landmarks model
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                             LANDMARKS_MODEL_URL, cache_subdir='temp'))
    
    # Initialize landmarks detector
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    
    # Create a temporary file for the aligned image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        aligned_face_path = tmp_file.name
    
    # Detect landmarks and align the face
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(image_path), start=1):
        print('Starting face alignment...')
        image_align(image_path, aligned_face_path, face_landmarks, 
                   output_size=output_size, x_scale=x_scale, 
                   y_scale=y_scale, em_scale=em_scale, alpha=alpha)
        print(f'Face aligned and saved to {aligned_face_path}')
        break  # Only process the first face
    
    return aligned_face_path

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

def load_model(model_path):
    """Load the trained face shape classifier model"""
    # First load a sample to get the input dimensions
    # We need to initialize the model with the same parameters
    
    # Determine input dimension from CLIP model
    # Using the same CLIP model as in extract_hidden_states.py
    #model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    #model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='metaclip_fullcc')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='metaclip_altogether')
    model = model.to(device)
    model.eval()
    
    # Get hidden state dimensions
    # For ViT-H-14, the hidden dimension will be different than ViT-B/16
    # Check the model's weight precision to match extract_hidden_states.py
    model_dtype = next(model.parameters()).dtype
    print(f"Model is using {model_dtype} precision")
    
    # Typical input dimension for ViT models
    input_dim = 1024 #1280 * 16 * 16  # Will be flattened in the classifier
    hidden_dim = 378#1024 
    num_classes = len(FACE_SHAPES)
    
    # Initialize the classifier with the same architecture
    classifier = FaceShapeClassifier(input_dim, hidden_dim, num_classes)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    return model, classifier, preprocess

def extract_clip_features(image_path, clip_model, preprocess):
    """Extract CLIP features from an image using standard forward pass"""
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')

        # Make image square with white background and resize
        image = make_square_with_white_background(image, target_size=244)
        
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Match the model precision
        model_dtype = next(clip_model.parameters()).dtype
        image_input = image_input.to(dtype=model_dtype)
        
        # Extract features using standard forward pass
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            
            # Flatten the features to match the classifier's expected input
            hidden_state = image_features.view(image_features.size(0), -1)
            
        return hidden_state
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def unload_model(clip_model, classifier_model):
    """Unload models and free up memory"""
    # Move models to CPU first
    clip_model.cpu()
    classifier_model.cpu()
    
    # Delete models
    del clip_model
    del classifier_model
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    # Check if file exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    # Align the face first
    print(f"Aligning face in image: {TEST_IMAGE_PATH}")
    aligned_image_path = TEST_IMAGE_PATH#align_face(TEST_IMAGE_PATH, output_size=512)
    
    # Load models
    print("Loading models...")
    clip_model, classifier_model, preprocess = load_model(MODEL_PATH)
    
    try:
        # Process the aligned image
        print(f"Processing aligned image: {aligned_image_path}")
        features = extract_clip_features(aligned_image_path, clip_model, preprocess)
            
        # Predict face shape
        with torch.no_grad():
            outputs = classifier_model(features)
            _, predicted = torch.max(outputs, 1)
            predicted_class_idx = predicted.item()
            predicted_class = FACE_SHAPES[predicted_class_idx]
        
        # Print predicted class
        print(f"Predicted face shape: {predicted_class}")
        
        # Print confidence scores for all classes
        softmax = nn.Softmax(dim=1)
        probabilities = softmax(outputs)
        
        print("\nConfidence scores:")
        for i, shape in enumerate(FACE_SHAPES):
            print(f"{shape}: {probabilities[0][i].item():.4f}")
    finally:
        # Always unload models, even if an error occurs
        print("Unloading models...")
        unload_model(clip_model, classifier_model)
    
    # Clean up temporary file
    #if os.path.exists(aligned_image_path):
    #    os.unlink(aligned_image_path)

if __name__ == "__main__":
    main() 