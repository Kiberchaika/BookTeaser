import os
import tempfile
import gradio as gr
import cv2
import numpy as np

# Set GPU 1 as visible
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from server_threaded import (
    download_all_models,
    initialize_sessions_and_globals,
    get_face_embedding,
    perform_face_swap_video_threaded
)

# Initialize the required models and sessions
print("Initializing models...")
if not download_all_models():
    print("One or more models failed to download. Exiting.")
    exit(1)

try:
    initialize_sessions_and_globals()
    print("Models initialized successfully.")
except Exception as e:
    print(f"Error initializing models: {e}")
    exit(1)

# Define face shapes and get face images from shape_faces_test directory
FACE_SHAPES = ["1", "2", "3", "4", "5"]
GENDERS = ["Мужчина", "Женщина"]

def get_face_images_by_shape(base_dir="shape_faces_test"):
    """Get all face images organized by gender and shape"""
    faces = {}
    for gender in GENDERS:
        faces[gender] = {}
        for shape in FACE_SHAPES:
            shape_dir = os.path.join(base_dir, gender, shape)
            if os.path.exists(shape_dir):
                images = []
                for file in os.listdir(shape_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append(os.path.join(shape_dir, file))
                faces[gender][shape] = images
            else:
                faces[gender][shape] = []
    return faces

FACE_IMAGES = get_face_images_by_shape()

def process_video(video_input, image_input, num_threads=4, progress=gr.Progress()):
    """
    Process the input video using the source image for face swapping.
    
    Args:
        video_input: Path to the input video
        image_input: Path to the source image
        num_threads: Number of threads to use for processing
        progress: Gradio progress indicator
    
    Returns:
        Tuple of (processed video path, status message)
    """
    # Create temp directory to store output
    temp_output_dir = tempfile.mkdtemp()
    output_video_path = os.path.join(temp_output_dir, "processed_video.mp4")
    
    # Read the source image
    progress(0, desc="Чтение исходного изображения...")
    source_img_bgr = cv2.imread(image_input)
    if source_img_bgr is None:
        return None, "Не удалось прочитать исходное изображение"
    
    # Get face embedding from source image
    progress(0.1, desc="Обнаружение лица в исходном изображении...")
    source_embedding = get_face_embedding(source_img_bgr)
    if source_embedding is None:
        return None, "Лицо не обнаружено в исходном изображении"
    
    # Create a dynamic status message that will be updated during processing
    status_message = "Обработка видео..."
    
    # Define a callback function to update progress
    def update_progress(value, description):
        nonlocal status_message
        # Store latest status message
        status_message = description
        
        # Map the 0-1 range to our desired range (0.2-1.0)
        # We use 0.2 as starting point since 0.0-0.2 is already used for prep steps
        progress_value = 0.2 + (value * 0.8)
        progress(progress_value, desc=description)
    
    # Process the video
    progress(0.2, desc="Начало обработки видео...")
    
    success = perform_face_swap_video_threaded(
        source_embedding,
        video_input,
        output_video_path,
        temp_dir_base=temp_output_dir,
        cache_dir=None,
        num_threads=num_threads,
        progress_callback=update_progress
    )
    
    if not success:
        return None, f"Не удалось обработать видео: {status_message}"
    
    progress(1.0, desc="Обработка завершена!")
    return output_video_path, f"Обработка видео успешно завершена!"

def update_images(gender, face_shape):
    """Update the gallery images based on selected gender and face shape"""
    if gender and face_shape and gender in FACE_IMAGES and face_shape in FACE_IMAGES[gender]:
        images = FACE_IMAGES[gender][face_shape]
        # Return both the images for the gallery (as list of image paths) and the raw image paths for the state
        return images, images
    return [], []

def select_image(evt: gr.SelectData, images):
    """Select an image from the gallery"""
    if images and 0 <= evt.index < len(images):
        return images[evt.index]
    return None

def process_video_wrapper(video_input, image_input, num_threads=4, progress=gr.Progress()):
    """
    Wrapper for process_video that handles button state management.
    """
    try:
        # Process the video and return results
        result = process_video(video_input, image_input, num_threads, progress)
        # Return results only (button state managed by event handler)
        return result
    except Exception as e:
        # If there's an error, return error message
        return None, f"Error: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Замена лица") as app:
    gr.Markdown("# Тест замены лица")
    gr.Markdown("Загрузите видео и выберите лицо для замены")
    
    # Hidden state for storing current images displayed in gallery
    current_images = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Входное видео")
            
            with gr.Tabs():
                with gr.TabItem("Выбрать лицо"):
                    with gr.Row():
                        gender_dropdown = gr.Dropdown(
                            choices=GENDERS,
                            label="Пол",
                            value="Мужчина"
                        )
                        shape_dropdown = gr.Dropdown(
                            choices=FACE_SHAPES,
                            label="Тип лица",
                            value="1"
                        )
                    
                    face_gallery = gr.Gallery(
                        label="Выберите лицо",
                        columns=3,
                        object_fit="contain",
                        height="auto"
                    )
                    
                    image_path = gr.Textbox(label="Selected Image Path", visible=False)
                    
                with gr.TabItem("Загрузить свое изображение"):
                    image_input = gr.Image(label="Исходное изображение (лицо для использования)", type="filepath")
            
            # Add interactive=False to prevent multiple clicks during processing
            submit_btn = gr.Button("Обработать видео", interactive=True, variant="primary")
        
        with gr.Column(scale=1):
            video_output = gr.Video(label="Обработанное видео")
            status_text = gr.Textbox(label="Статус обработки", value="Готов к обработке")
    
    # Set up event handlers
    gender_dropdown.change(
        fn=update_images,
        inputs=[gender_dropdown, shape_dropdown],
        outputs=[face_gallery, current_images]
    )
    
    shape_dropdown.change(
        fn=update_images,
        inputs=[gender_dropdown, shape_dropdown],
        outputs=[face_gallery, current_images]
    )
    
    face_gallery.select(
        fn=select_image,
        inputs=[current_images],
        outputs=[image_path]
    )
    
    # Load initial images when app starts
    app.load(
        fn=update_images,
        inputs=[gender_dropdown, shape_dropdown],
        outputs=[face_gallery, current_images]
    )
    
    # Process button handling for both tabs with progress tracking
    submit_btn.click(
        fn=lambda video, img_path, img_upload: process_video_wrapper(
            video, 
            img_path if img_path else img_upload, 
            4  # Fixed number of threads
        ),
        inputs=[video_input, image_path, image_input],
        outputs=[video_output, status_text],
        show_progress=True  # Enable progress bar
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7779, share=True) 