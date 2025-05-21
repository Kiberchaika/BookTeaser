<script>
    import { onMount, onDestroy, createEventDispatcher } from 'svelte';
    import * as faceapi from 'face-api.js';
    import { faceDetectionState, loadFaceDetectionModels } from '$lib/stores/faceDetectionStore';
  
    const dispatch = createEventDispatcher();
    // State variables
    let webcamRef;
    let canvasRef;
    let isVideoReady = false;
    let faceCount = 0;
    let debugInfo = "";
    let isShowDebugInfo = true;
    let stream;
    let hasFaceNearCenter = false;
    let isStopped = false;
    
    // Local reference to the face detection state
    let isModelLoaded = false;
    let modelError = null;
    
    // Export method to stop webcam
    export function stopWebcam() {
        isStopped = true;
        if (detectionFrameId) {
            cancelAnimationFrame(detectionFrameId);
        }
        if (renderFrameId) {//
            cancelAnimationFrame(renderFrameId);
        }
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    }
    
    // Subscribe to face detection state
    const unsubscribe = faceDetectionState.subscribe(state => {
      isModelLoaded = state.isLoaded;
      modelError = state.error;
      
      if (state.isLoaded && isVideoReady) {
        // Start face detection when models are loaded and video is ready
        startFaceDetection();
      }
      
      if (state.error) {
        debugInfo = `Model error: ${state.error}`;
      }
    });
    
    // Constants
    const DETECTION_SCALE = 0.5; // Scale factor for detection (0.25 = 25% of original size)
    const FPS_LIMIT = 15; // Limit detection to 10 frames per second
    const canvasWidth = 1920;
    const ROTATION_ANGLE = 90; // Rotation angle in degrees
    const CROP_PIXELS = 800; // How many pixels to crop from top and bottom
    
    // Store detection results to be used by render loop
    let currentDetections = [];
    let lastDetectionTime = 0;
    let canvasHeight = canvasWidth;  // Will be adjusted based on video dimensions
    
    onMount(async () => {
      // Set up webcam first
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1920  },
            height: { ideal: 1080 }
          },
          audio: false
        });
        
        if (webcamRef) {
          webcamRef.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing webcam:", error);
        debugInfo = `Error accessing webcam: ${error}`;
      }
      
      // Load face detection models from the store
      debugInfo = "Loading models...";
      await loadFaceDetectionModels();
      
      return () => {
        if (stream) {
          stream.getTracks().forEach(track => track.stop());
        }
        unsubscribe(); // Clean up subscription
      };
    });
    
    // Handle video being ready
    function handleVideoReady() {
      isVideoReady = true;
      debugInfo = "Video ready";
      startRenderLoop();
      
      // Only start face detection if models are loaded
      if (isModelLoaded) {
        startFaceDetection();
      }
    }
    
    let detectionFrameId;
    let renderFrameId;
    let isDetectionRunning = false;
    
    // Start face detection loop
    function startFaceDetection() {
      if (!isVideoReady || !isModelLoaded) {
        debugInfo = "Waiting for video and models to be ready...";
        return;
      }
      
      debugInfo = "Starting face detection...";
      detectFaces();
    }
    
    // Start separate render loop
    function startRenderLoop() {
      if (!isVideoReady) return;
      
      renderFrame();
    }
    
    // Render function - runs every frame
    function renderFrame() {
      if (isStopped || !webcamRef || !canvasRef) {
        renderFrameId = requestAnimationFrame(renderFrame);
        return;
      }
      
      const video = webcamRef;
      const canvas = canvasRef;
      
      // Make sure video is ready
      if (video.readyState !== 4) {
        renderFrameId = requestAnimationFrame(renderFrame);
        return;
      }
      
      // Calculate dimensions for rotated video
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;
      
      // For 90 degree rotation, we need to adjust canvas dimensions
      // to maintain the video aspect ratio after rotation, with cropping
      const rotatedAspectRatio = videoWidth / videoHeight;
      const fullCanvasHeight = Math.round(canvasWidth * rotatedAspectRatio);
      // Apply cropping to the height by removing CROP_PIXELS from top and bottom
      canvasHeight = Math.round(fullCanvasHeight - (2 * CROP_PIXELS));
      
      // Update canvas dimensions if needed
      if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
      }
      
      // Get canvas context
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        renderFrameId = requestAnimationFrame(renderFrame);
        return;
      }
      
      // Clear the canvas
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      
      // Save the current context state
      ctx.save();
      
      // Move to center, rotate, then move back
      ctx.translate(canvasWidth/2, canvasHeight/2);
      ctx.rotate(ROTATION_ANGLE * Math.PI / 180);
      ctx.translate(-canvasHeight/2, -canvasWidth/2);
      
      // Calculate scaling to fit the rotated video
      const scale = Math.min(canvasWidth / videoHeight, fullCanvasHeight / videoWidth);
      const scaledWidth = videoWidth * scale;
      const scaledHeight = videoHeight * scale;
      
      // Center the video and adjust for cropping
      const x = (canvasHeight - scaledWidth) / 2;
      const y = ((canvasWidth - scaledHeight) / 2) ;
      // Draw the video
      ctx.drawImage(video, x, y, scaledWidth, scaledHeight);
      
      // Restore the context state before drawing detections
      ctx.restore();
      
      // Adjust face detection coordinates to account for cropping
      const cropOffset = (fullCanvasHeight - canvasHeight) / 2;
      
      // Draw face detection results if available
      if (currentDetections.length > 0) {
        currentDetections.forEach((detection) => {
          if (!detection || !detection.detection || !detection.detection.box) return;
          
          const box = detection.detection.box;
          
          // Adjust box coordinates for cropping
          const adjustedBox = {
            x: box.x,
            y: box.y - cropOffset,
            width: box.width,
            height: box.height
          };

          // Only draw if the face is within the cropped area
          if (adjustedBox.y + adjustedBox.height > 0 && adjustedBox.y < canvasHeight) {
            // Draw rectangle around face only when debug info is shown
            if (isShowDebugInfo) {
              ctx.strokeStyle = "#00ff00";
              ctx.lineWidth = 3;
              ctx.strokeRect(adjustedBox.x, adjustedBox.y, adjustedBox.width, adjustedBox.height);
              
              // Add label if it would be visible
              if (adjustedBox.y > 25) {
                ctx.fillStyle = "#00ff00";
                ctx.fillRect(adjustedBox.x, adjustedBox.y - 25, 70, 25);
                ctx.fillStyle = "#000000";
                ctx.font = "16px Arial";
                ctx.fillText("Face", adjustedBox.x + 5, adjustedBox.y - 7);
              }
            }
            
            // Visualize facial landmarks if enabled and landmarks exist
            if (isShowDebugInfo && detection.landmarks && detection.landmarks.positions) {
              const landmarks = detection.landmarks;
              ctx.fillStyle = "#ffffff";
              landmarks.positions.forEach((point) => {
                // Adjust landmark coordinates for cropping
                const adjustedY = point.y - cropOffset;
                // Only draw landmarks that are within the cropped area
                if (adjustedY > 0 && adjustedY < canvasHeight) {
                  ctx.beginPath();
                  ctx.arc(point.x, adjustedY, 2, 0, 2 * Math.PI);
                  ctx.fill();
                }
              });
            }
          }
        });
      }
      
      // Draw debug info directly on canvas
      if (isShowDebugInfo) {
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(10, canvasHeight - 145, 400, 135);
        
        ctx.fillStyle = "white";
        ctx.font = "16px Arial";
        ctx.fillText(`Status: ${isModelLoaded ? "Model loaded" : "Loading model..."} | ${isVideoReady ? "Video ready" : "Waiting for video..."}`, 20, canvasHeight - 115);
        ctx.fillText(`Faces detected: ${faceCount}${hasFaceNearCenter ? ' (Face near center!)' : ''}`, 20, canvasHeight - 90);
        ctx.fillText(`Debug: ${debugInfo}`, 20, canvasHeight - 65);
        ctx.fillText(`Canvas: ${canvasWidth} × ${canvasHeight} px (Crop: ${Math.round(CROP_PIXELS / fullCanvasHeight * 100)}%)`, 20, canvasHeight - 40);
        ctx.fillText(`Press SPACE to toggle landmark visualization`, 20, canvasHeight - 15);
      }
      
      // Continue render loop
      renderFrameId = requestAnimationFrame(renderFrame);
    }
    
    // Add this function before detectFaces()
    function findMostCenteredFace(faceBoxes) {
        if (!webcamRef || !webcamRef.videoWidth || !webcamRef.videoHeight || !faceBoxes || faceBoxes.length === 0) {
            return { isNearCenter: false, faceBox: null, faceCrop: null };
        }

        const videoWidth = webcamRef.videoWidth;
        const videoHeight = webcamRef.videoHeight;
        const aspectRatio = videoHeight / videoWidth;
        const canvasHeight = Math.round(canvasWidth * aspectRatio);

        const centerX = canvasWidth / 2;
        const allowedRange = 300; // 300px range from center
        
        // Calculate distances for all faces and find the most centered one
        const facesWithDistances = faceBoxes.map(box => {
            const faceCenterX = box.x + (box.width / 2);
            const distanceX = Math.abs(faceCenterX - centerX);
            
            return {
                box,
                distance: distanceX,
                isNearCenter: distanceX <= allowedRange
            };
        });

        // Sort by distance and get the closest face
        const closestFace = facesWithDistances.sort((a, b) => a.distance - b.distance)[0];

        // Create a square crop of the face
        let faceCrop = null;
        if (closestFace && canvasRef) {
            const box = closestFace.box;
            const size = Math.max(box.width, box.height) * 1.5; // Increased by 50%
            const x = Math.max(0, box.x - (size - box.width) / 2);
            const y = Math.max(0, box.y - (size - box.height) / 2);
            
            // Create temporary canvas for cropping
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = size;
            tempCanvas.height = size;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Draw the cropped face
            tempCtx.drawImage(
                canvasRef,
                x, y, size, size,  // Source rectangle
                0, 0, size, size   // Destination rectangle
            );
            
            faceCrop = tempCanvas;
        }

        return {
            isNearCenter: closestFace.isNearCenter,
            faceBox: closestFace.box,
            faceCrop: faceCrop
        };
    }
    
    // Face detection function - runs at limited FPS
    async function detectFaces() {
      if (isStopped || !webcamRef || !isModelLoaded || !isVideoReady) {
        detectionFrameId = requestAnimationFrame(detectFaces);
        return;
      }
      
      if (isDetectionRunning) {
        detectionFrameId = requestAnimationFrame(detectFaces);
        return;
      }
      
      // Apply FPS limiting
      const now = performance.now();
      const elapsed = now - lastDetectionTime;
      const fpsInterval = 1000 / FPS_LIMIT;
      
      if (elapsed < fpsInterval) {
        detectionFrameId = requestAnimationFrame(detectFaces);
        return;
      }
      
      lastDetectionTime = now - (elapsed % fpsInterval);
      isDetectionRunning = true;
      
      try {
        const video = webcamRef;
        
        // Make sure video is ready
        if (video.readyState !== 4) {
          isDetectionRunning = false;
          detectionFrameId = requestAnimationFrame(detectFaces);
          return;
        }
        
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;
        
        // Create a temporary canvas for detection with rotated dimensions
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        // Set dimensions for detection (swap width/height for rotation)
        tempCanvas.width = canvasWidth * DETECTION_SCALE;
        tempCanvas.height = (canvasWidth * (videoWidth / videoHeight)) * DETECTION_SCALE;
        
        // Draw rotated video to temp canvas
        tempCtx.translate(tempCanvas.width/2, tempCanvas.height/2);
        tempCtx.rotate(ROTATION_ANGLE * Math.PI / 180);
        tempCtx.translate(-tempCanvas.height/2, -tempCanvas.width/2);
        
        // Draw scaled down version of video to temp canvas
        const scale = Math.max(tempCanvas.height / videoWidth, tempCanvas.width / videoHeight);
        const scaledWidth = videoWidth * scale;
        const scaledHeight = videoHeight * scale;
        const x = (tempCanvas.height - scaledWidth) / 2;
        const y = (tempCanvas.width - scaledHeight) / 2;
        
        tempCtx.drawImage(video, x, y, scaledWidth, scaledHeight);
        
        // Double-check models are loaded before detection
        if (!faceapi.nets.ssdMobilenetv1.isLoaded || !faceapi.nets.faceLandmark68Net.isLoaded) {
          console.warn("Models not loaded yet, skipping detection");
          isDetectionRunning = false;
          detectionFrameId = requestAnimationFrame(detectFaces);
          return;
        }
        
        // Detect faces with landmarks on the smaller canvas
        const detections = await faceapi.detectAllFaces(tempCanvas).withFaceLandmarks();
        faceCount = detections.length;
        
        // Scale the detections back to original size
        const scaledDetections = detections.map(detection => {
          // Scale the detection box
          const scaledBox = {
            x: detection.detection.box.x / DETECTION_SCALE,
            y: detection.detection.box.y / DETECTION_SCALE,
            width: detection.detection.box.width / DETECTION_SCALE,
            height: detection.detection.box.height / DETECTION_SCALE
          };
          
          // Scale the landmarks
          const scaledLandmarks = {
            positions: detection.landmarks.positions.map(point => ({
              x: point.x / DETECTION_SCALE,
              y: point.y / DETECTION_SCALE
            }))
          };
          
          return {
            detection: { box: scaledBox },
            landmarks: scaledLandmarks
          };
        });
        
        // Update the current detections for rendering
        currentDetections = scaledDetections;
        
        // Check if any face is near center
        const centeredFaceResult = findMostCenteredFace(scaledDetections.map(d => d.detection.box));
        hasFaceNearCenter = centeredFaceResult.isNearCenter;
        
        if (scaledDetections.length === 0) {
          debugInfo = "No faces detected";
        } else {
          debugInfo = `${scaledDetections.length} faces detected (processing at ${Math.round(tempCanvas.width)}x${Math.round(tempCanvas.height)}, ${FPS_LIMIT} FPS)${hasFaceNearCenter ? ' - Face near center!' : ''}`;
        }
        
        // Clean up temp canvas
        tempCanvas.remove();
      } catch (error) {
        console.error("Error in face detection:", error);
        debugInfo = `Error in detection: ${error}`;
      } finally {
        isDetectionRunning = false;
        // Continue detection loop
        detectionFrameId = requestAnimationFrame(detectFaces);
      }
    }
    
    onDestroy(() => {
      if (detectionFrameId) {
        cancelAnimationFrame(detectionFrameId);
      }
      
      if (renderFrameId) {
        cancelAnimationFrame(renderFrameId);
      }
      
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    });

    // Update the hasFaceNearCenter state and dispatch event
    $: {
        if (hasFaceNearCenter !== undefined) {
            const centeredFaceResult = findMostCenteredFace(currentDetections.map(d => d.detection.box));
            dispatch('faceCenter', { 
                isNearCenter: hasFaceNearCenter,
                faceCrop: centeredFaceResult.faceCrop
            });
        }
    }
</script>

<style>
    .container {
        position: relative;
        width: 100vw;
        height: 100vh;
    }
    
    .loading-overlay {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: rgba(0, 0, 0, 0.7);
        z-index: 10;
        color: white;
        font-size: 1.25rem;
    }
    
    .video {
        width: 100%;
        height: 100%;
        position: absolute;
        opacity: 0;
    }
    
    .canvas {
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        object-fit: contain;
        margin: 0 auto;
        border-radius: 150px;
    }
</style>

<div class="container">
    {#if !isModelLoaded}
        <div class="loading-overlay">
        Loading face detection models...
        </div>
    {/if}

    <video
        bind:this={webcamRef}
        autoplay
        playsinline
        muted
        class="video"
        on:loadedmetadata={handleVideoReady}
    ></video>

    <canvas
        bind:this={canvasRef}
        class="canvas"
    ></canvas>
</div>