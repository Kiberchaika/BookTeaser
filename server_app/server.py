import asyncio
import websockets
import json
import base64
from datetime import datetime
import os

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
                    print(f"Selected Scene: {data['selectedScene']}")
                    print(f"Image saved as: {filename}")
                    
                    # Wait for 5 seconds
                    print("Waiting 5 seconds...")
                    await asyncio.sleep(5)
                    
                    # Send test URL back to client
                    test_url = "https://example.com/test-result"
                    await websocket.send(json.dumps({
                        "status": "success",
                        "message": "Face capture received and saved successfully",
                        "resultUrl": test_url
                    }))
                    print(f"Sent test URL: {test_url}")
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
    
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
