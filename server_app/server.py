import os
import sys
import asyncio
import websockets
import json
import base64
from datetime import datetime
import os
from face_shape_classifier.inference import initialize_models, preprocess_image, unload_model

from info import characters, shapes

                
# Create directory for saving images if it doesn't exist
SAVE_DIR = "received_images"
os.makedirs(SAVE_DIR, exist_ok=True)

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

                    



                    # Wait for 5 seconds
                    print("Waiting 5 seconds...")
                    await asyncio.sleep(5)
                    
                    # Send result back to client
                    await websocket.send(json.dumps({
                        "status": "success",
                        "message": "Face capture received and processed successfully",
                        "faceShape": face_shape,
                        "resultUrl": "https://example.com/test-result"
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
