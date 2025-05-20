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
    let isShowDebugInfo = false;
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
        if (renderFrameId) {
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
    
    // Store detection results to be used by render loop
    let currentDetections = [];
    let lastDetectionTime = 0;
    
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
      
      // Set canvas dimensions width and maintain aspect ratio
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;
      const aspectRatio = videoHeight / videoWidth;
      
      const canvasHeight = Math.round(canvasWidth * aspectRatio);
      
      // Ensure canvas has the correct dimensions
      if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
      }

      //console.log(canvasWidth, canvasHeight);
      
      // Get canvas context
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        renderFrameId = requestAnimationFrame(renderFrame);
        return;
      }
      
      // Draw video frame on canvas first
      ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);
      
      // Draw face detection results if available
      if (currentDetections.length > 0) {
        currentDetections.forEach((detection) => {
          if (!detection || !detection.detection || !detection.detection.box) return;
          
          const box = detection.detection.box;

          // Draw rectangle around face only when debug info is shown
          if (isShowDebugInfo) {
            ctx.strokeStyle = "#00ff00";
            ctx.lineWidth = 3;
            ctx.strokeRect(box.x, box.y, box.width, box.height);
            
            // Add label
            ctx.fillStyle = "#00ff00";
            ctx.fillRect(box.x, box.y - 25, 70, 25);
            ctx.fillStyle = "#000000";
            ctx.font = "16px Arial";
            ctx.fillText("Face", box.x + 5, box.y - 7);
          }
          
          // Visualize facial landmarks if enabled and landmarks exist
          if (isShowDebugInfo && detection.landmarks && detection.landmarks.positions) {
            const landmarks = detection.landmarks;
            const positions = landmarks.positions;
            
            // Draw all facial landmarks
            ctx.fillStyle = "#ffffff";
            
            // Draw facial landmarks
            positions.forEach((point) => {
              ctx.beginPath();
              ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
              ctx.fill();
            });
            
            // Optional: Draw connecting lines between landmarks for better visualization
            if (positions.length > 0) {
              // Draw outline of face
              ctx.strokeStyle = "#3366ff";
              ctx.lineWidth = 1;
              
              // Helper functions for drawing curves
              const drawCurve = (points) => {
                if (points.length === 0) return;
                
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                
                for (let i = 1; i < points.length; i++) {
                  ctx.lineTo(points[i].x, points[i].y);
                }
                
                ctx.stroke();
              };
              
              const drawClosedCurve = (points) => {
                if (points.length === 0) return;
                
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                
                for (let i = 1; i < points.length; i++) {
                  ctx.lineTo(points[i].x, points[i].y);
                }
                
                ctx.closePath();
                ctx.stroke();
              };
              
              // Draw facial features
              // Draw jaw line
              drawCurve(positions.slice(0, 17));
              
              // Draw eyebrows
              drawCurve(positions.slice(17, 22));
              drawCurve(positions.slice(22, 27));
              
              // Draw nose
              drawCurve(positions.slice(27, 36));
              
              // Draw eyes
              drawClosedCurve(positions.slice(36, 42));
              drawClosedCurve(positions.slice(42, 48));
              
              // Draw mouth
              drawClosedCurve(positions.slice(48, 60));
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
        ctx.fillText(`Canvas: ${canvasWidth} × ${canvasHeight} px`, 20, canvasHeight - 40);
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
      const centerY = canvasHeight / 2;
      
      const maxDistanceX = canvasWidth / 8;
      const maxDistanceY = canvasHeight / 8;

      // Calculate distances for all faces and find the most centered one
      const facesWithDistances = faceBoxes.map(box => {
        const faceCenterX = box.x + (box.width / 2);
        const faceCenterY = box.y + (box.height / 2);
        
        const distanceX = Math.abs(faceCenterX - centerX);
        const distanceY = Math.abs(faceCenterY - centerY);
        
        const totalDistance = Math.sqrt(distanceX * distanceX + distanceY * distanceY);
        
        return {
          box,
          distance: totalDistance,
          isNearCenter: distanceX <= maxDistanceX && distanceY <= maxDistanceY
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
        const aspectRatio = videoHeight / videoWidth;
        
        const canvasHeight = Math.round(canvasWidth * aspectRatio);
        
        // Create a smaller temporary canvas for detection
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        // Set smaller dimensions for detection
        tempCanvas.width = canvasWidth * DETECTION_SCALE;
        tempCanvas.height = canvasHeight * DETECTION_SCALE;
        
        // Draw scaled down version of video to temp canvas
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        
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