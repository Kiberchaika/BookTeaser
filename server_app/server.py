import os
import sys
import asyncio
import websockets
import json
import base64
from datetime import datetime
import os
import cv2
import subprocess
from face_shape_classifier.inference import initialize_models, preprocess_image, unload_model
from facefusion.inference_threaded import (
    initialize_sessions_and_globals,
    get_source_face_embedding,
    perform_face_swap_video_threaded,
    unload_models_and_clear_memory
)

from info import characters, shapes

# Create directories for saving images and videos if they don't exist
SAVE_DIR = "received_images"
TMP_DIR = "tmp"
STORAGE_DIR = "storage"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

async def process_videos(face_image_path, character_info, face_shape_index):
    os.makedirs(TMP_DIR, exist_ok=True)
    
    # Initialize face fusion models
    initialize_sessions_and_globals()
    
    # Get source face embedding from captured image
    source_img_bgr = cv2.imread(face_image_path)
    if source_img_bgr is None:
        print(f"Error: Could not load source image from {face_image_path}")
        return False
    
    source_arcface_embedding = get_source_face_embedding(source_img_bgr)
    if source_arcface_embedding is None:
        print("Could not get source face embedding. Exiting.")
        return False

    character_folder = character_info["folder"]
    swapper_model = character_info["swapper"]
    
    # Process each scene
    for scene_num, scene_info in character_info["scenes"].items():
        scene_folder = f"videos_processed/{character_folder}/scene{scene_num}"
        
        if not scene_info.get("replace_face", False):
            # Just copy the video.mp3 and N.mp4 to tmp folder
            input_video = f"{scene_folder}/video.mp4"
            output_video = f"{TMP_DIR}/{scene_num}.mp4"
            if os.path.exists(input_video):
                subprocess.run(["cp", input_video, output_video])
            continue
            
        # For scenes with replace_face=True, process only the face_shape_index video
        protect_eyes = scene_info.get("protect_eyes", False)
        input_video = f"{scene_folder}/{face_shape_index}.mp4"
        output_video = f"{TMP_DIR}/{scene_num}.mp4"
        cache_dir = f"{scene_folder}/{face_shape_index}"
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"\nProcessing {character_folder} - Scene {scene_num}")
        print(f"Using {swapper_model} swapper, protect_eyes={protect_eyes}")
        
        # Process the video
        success = perform_face_swap_video_threaded(
            source_arcface_embedding,
            input_video,
            output_video,
            swapper_model_name=swapper_model,
            use_eye_mask=protect_eyes,
            cache_dir=cache_dir,
            num_threads=4
        )
        
        if not success:
            print(f"Failed to process scene {scene_num}")
            return False

    # Join all videos using ffmpeg
    scene_files = [f"{i}.mp4" for i in range(1, len(character_info["scenes"]) + 1)]
    scene_files_str = " ".join(scene_files)
    
    # Create a file list for ffmpeg
    with open(f"{TMP_DIR}/filelist.txt", "w") as f:
        for file in scene_files:
            f.write(f"file '{file}'\n")
    
    # Generate timestamp with milliseconds
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}.mp4"
    output_path = f"{STORAGE_DIR}/{output_filename}"
    
    # Use ffmpeg to concatenate videos
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", f"{TMP_DIR}/filelist.txt",
        "-c", "copy",
        output_path
    ])

    # delete tmp folder
    subprocess.run(["rm", "-rf", TMP_DIR])
    
    # Cleanup
    unload_models_and_clear_memory()
    return output_filename

async def handle_websocket(websocket):
    try:
        print(f"New client connected from {websocket.remote_address}")
        async for message in websocket:
            try:
                # Parse the JSON message
                data = json.loads(message)
                
                if data['type'] == 'process_face':
                    # Extract the base64 image data (remove the data URL prefix)
                    image_data = data['face_image'].split(',')[1]
                    
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data)
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{SAVE_DIR}/face_capture_{timestamp}.jpg"
                    
                    # Save the image
                    with open(filename, 'wb') as f:
                        f.write(image_bytes)
                    
                    # Log the received data
                    print(f"Received face capture:")
                    print(f"Gender: {data['gender']}")
                    print(f"Selected Character: {data['character']}")
                    print(f"Image saved as: {filename}")


                    # Initialize models at startup
                    initialize_models()

                    # Get face shape classification
                    face_shape = preprocess_image(filename)
                    print(f"Predicted face shape: {face_shape}")
                    
                    # Clean up models when server shuts down
                    unload_model()


                    # Get character info
                    character_info = characters[data['character']]
                    print(f"Selected character: {character_info}")  

                    # find index of face_shape in shapes
                    face_shape_index = list(shapes.keys()).index(face_shape)
                    print(f"Face shape index: {face_shape_index}")

                    # Process videos
                    success = await process_videos(filename, character_info, face_shape_index + 1)  # +1 because face_shape_index is 0-based
                    if not success:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": "Failed to process videos"
                        }))
                        continue

                    # Wait for 5 seconds
                    print("Waiting 5 seconds...")
                    await asyncio.sleep(5)
                    
                    # Send result back to client
                    await websocket.send(json.dumps({
                        "status": "success",
                        "message": "Face capture received and processed successfully",
                        "faceShape": face_shape,
                        "resultUrl": f"{STORAGE_DIR}/{success}"  # success contains the output filename
                    }))
                    print(f"Sent face shape result: {face_shape}")
                elif data['type'] == 'test_connection':
                    # Handle test connection message
                    print(f"Received test connection from client")
                    await websocket.send(json.dumps({
                        "status": "success",
                        "message": "Test connection successful"
                    }))
                    
            except json.JSONDecodeError:
                print("Error: Invalid JSON received")
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": f"Error processing message: {str(e)}"
                }))
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client disconnected: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

async def main():
    # Start WebSocket server with CORS support
    server = await websockets.serve(
        handle_websocket,
        "0.0.0.0",
        7779,
        ping_interval=None,  # Disable ping to prevent connection issues
        ping_timeout=None,
        close_timeout=None,
        max_size=None,  # No limit on message size
        max_queue=None,  # No limit on message queue
        compression=None  # Disable compression for better performance
    )
    
    print("WebSocket server started")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
