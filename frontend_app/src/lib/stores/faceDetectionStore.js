import { writable } from 'svelte/store';
import * as faceapi from 'face-api.js';

// Face detection model state
export const faceDetectionState = writable({
    isLoading: false,
    isLoaded: false,
    error: null
});

// Constants
const MODEL_URL = "/weights";

// Function to load face detection models
export async function loadFaceDetectionModels() {
    // Check if models are already loaded
    if (faceapi.nets.ssdMobilenetv1.isLoaded && faceapi.nets.faceLandmark68Net.isLoaded) {
        faceDetectionState.set({ isLoading: false, isLoaded: true, error: null });
        return true;
    }
    
    // Set loading state
    faceDetectionState.set({ isLoading: true, isLoaded: false, error: null });
    
    try {
        // Load face detection models
        await Promise.all([
            faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
            faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL)
        ]);

        await testFaceDetectionOnBlackCanvas();
        
        console.log("Face detection and landmark models loaded successfully");
        faceDetectionState.set({ isLoading: false, isLoaded: true, error: null });
        return true;
    } catch (error) {
        console.error("Error loading face detection models:", error);
        faceDetectionState.set({ isLoading: false, isLoaded: false, error: error.message });
        return false;
    }
} 

// Test face detection on black canvas
async function testFaceDetectionOnBlackCanvas() {
    // Create a temporary 100x100 black canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 100;
    tempCanvas.height = 100;
    
    // Get the 2D context and fill with black
    const ctx = tempCanvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 100, 100);
    
    try {
        // Ensure models are loaded
        if (!faceapi.nets.ssdMobilenetv1.isLoaded || !faceapi.nets.faceLandmark68Net.isLoaded) {
            return null;
        }
        
        // Run face detection on the black canvas
        const detections = await faceapi.detectAllFaces(tempCanvas).withFaceLandmarks();
        console.log('Face detection results on black canvas:', detections);
        return detections;
    } catch (error) {
        console.error('Error during test face detection:', error);
        return null;
    }
} 