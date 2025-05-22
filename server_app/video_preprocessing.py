from info import characters
import os
import cv2
from facefusion.inference_threaded import (
    initialize_sessions_and_globals,
    get_source_face_embedding,
    perform_face_swap_video_threaded,
    unload_models_and_clear_memory
)

def process_videos():
    # Initialize face fusion models
    initialize_sessions_and_globals()
    
    # Get source face embedding from test image
    source_img_path = "server_app/face_shape_classifier/test.jpg"
    source_img_bgr = cv2.imread(source_img_path)
    if source_img_bgr is None:
        print(f"Error: Could not load source image from {source_img_path}")
        return
    
    source_arcface_embedding = get_source_face_embedding(source_img_bgr)
    if source_arcface_embedding is None:
        print("Could not get source face embedding. Exiting.")
        return

    # Process each character's videos
    for character_id, character_info in characters.items():
        character_folder = character_info["folder"]
        swapper_model = character_info["swapper"]
        
        # Process each scene
        for scene_num, scene_info in character_info["scenes"].items():
            if not scene_info.get("replace_face", False):
                continue
                
            protect_eyes = scene_info.get("protect_eyes", False)
            
            # Construct paths
            scene_folder = f"videos_processed/{character_folder}/scene{scene_num}"


            # Interate face from 1 to 5
            for face_num in range(1, 6):
                input_video = f"{scene_folder}/{face_num}.mp4"
                output_video = f"tmp.mp4"
                cache_dir = f"{scene_folder}/{face_num}"
                
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
                    num_threads=4,  # Adjust based on your CPU cores
                    use_enhance=False
                )
                
                if success:
                    print(f"Successfully processed scene {scene_num}")
                else:
                    print(f"Failed to process scene {scene_num}")

    # Cleanup
    unload_models_and_clear_memory()

if __name__ == "__main__":
    process_videos()