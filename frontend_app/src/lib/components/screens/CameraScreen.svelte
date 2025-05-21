<script>
    import Webcam from '$lib/components/Webcam.svelte';
    import { appState, navigateTo, navigateBack, wsConnection } from '$lib/stores/appStore.js';
    import CircleLinearProgressBar from '$lib/components/CircleLinearProgressBar.svelte';
    import CircularProgressBar from '$lib/components/CircularProgressBar.svelte';
    import Counter from '$lib/components/Counter.svelte';
    import { onMount } from 'svelte';
    
    let currentState = 0; // 0: initial, 1: second state, 2: third state
    let progress = 0;
    let circularProgress = 0;
    let animationFrameId;
    let counterComponent;
    let isFaceNearCenter = false;
    let webcamComponent;
    let faceCrop = null;
    
    $: characterIconStyle = currentState === 0 ? 'opacity: 0;' : 'opacity: 1;';
    $: stopBtnStyle = currentState !== 2 ? '' : 'display: none;';
    $: startBtnStyle = currentState === 0 ? `transform: translateX(-50%) scale(1);` : 'display: none;';
    $: tipsStyle = currentState === 0 ? '' : 'display: none;';
    $: progressCircularStyle = currentState === 2 ? '' : 'display: none;';
    $: backgroundImageStyle = `background-image: url('screens/camera/text${currentState + 1}.png');`;
    $: characterIconSrc = $appState.userSettings.gender === 'female' 
        ? `screens/selectionWoman/${$appState.userSettings.character}.png`
        : `screens/selectionMan/${$appState.userSettings.character}.png`;
    
    // Start counter when currentState becomes 1
    $: if (currentState === 1 && counterComponent) {
        counterComponent.startCounter();
    }
    
    function lerp(start, end, factor) {
        return start + (end - start) * factor;
    }
    
    function animateProgress() {
        const targetProgress = (currentState + 1) / 3;
        progress = lerp(progress, targetProgress, 0.005); // Adjust factor for speed
        
        if (Math.abs(progress - targetProgress) > 0.001) {
            animationFrameId = requestAnimationFrame(animateProgress);
        }
    }
    
    $: {
        if (currentState !== undefined) {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
            animateProgress();
        }
    }
    
    onMount(() => {
        return () => {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
        };
    });
    
    function handleCounterComplete() {
        console.log("handleCounterComplete");
        if (webcamComponent) {
            webcamComponent.stopWebcam();
        }

        // Save face crop if available
        if (faceCrop) {
            try {
                // Convert canvas to data URL
                const dataUrl = faceCrop.toDataURL('image/jpeg', 0.95);
                

                 
                // Create a download link
                const link = document.createElement('a');
                link.download = `face_capture_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
                link.href = dataUrl;
                
                // Trigger download
                document.body.appendChild(link);
                //link.click();
                document.body.removeChild(link);
                
                console.log("Face crop saved successfully");
                
                // Send face crop and user settings via WebSocket
                const ws = $wsConnection;
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const message = {
                        type: 'process_face',
                        face_image: dataUrl,
                        gender: $appState.userSettings.gender,
                        character: $appState.userSettings.character
                    };
                    ws.send(JSON.stringify(message));
                    console.log("Face crop and settings sent via WebSocket");
                } else {
                    console.error("WebSocket is not connected");
                }
            } catch (error) {
                console.error("Error sending face crop:", error);
            }

            currentState = 2;
            animateCircularProgress();
        }
        else {
            // goto welcome screen
            navigateTo('welcome');
        }
    }

    function handleStartClick() {
        currentState = 1;
        faceCrop = null;
    }

    function handleCharacterSelect(character) {
        appState.update(state => ({
            ...state,
            userSettings: {
                ...state.userSettings,
                character: character
            }
        }));
        
        navigateTo('camera');
    }

    function handleWomanClick() {
        appState.update(state => ({
            ...state,
            userSettings: {
                ...state.userSettings,
                gender: "female"
            }
        }));
        navigateTo('scene-selection-woman');
    }

    function animateCircularProgress() {
        const startTime = performance.now();
        const duration = 3000; // 3 seconds

        function updateProgress(currentTime) {
            const elapsed = currentTime - startTime;
            circularProgress = Math.min(elapsed / duration, 1);

            if (circularProgress < 1) {
                requestAnimationFrame(updateProgress);
            } else {
                navigateTo('progress');
            }
        }

        requestAnimationFrame(updateProgress);
    }

    function handleFaceCenter(event) {
        isFaceNearCenter = event.detail.isNearCenter;
        faceCrop = event.detail.faceCrop;

        // If face was near center but is no longer, reset state and replay animations
        if (currentState !== 0 && !isFaceNearCenter) {
            return; // temporary method to prevent reset, don't change this line

            currentState = 0;
            faceCrop = null;

            // Reset counter if it exists
            if (counterComponent) {
                counterComponent.resetCounter();
            }

            // Force re-render of tips by temporarily removing and re-adding them
            const tips = document.querySelectorAll('.tip-icon');
            tips.forEach(tip => {
                tip.style.animation = 'none';
                tip.offsetHeight; // Trigger reflow
                tip.style.animation = null;
            });
        }
    }
</script>

<div class="screen-container">
    <video autoplay muted loop playsinline class="background-video">
        <source src="screens/camera/background.mp4" type="video/mp4">
    </video>
    <div class="background-image" style={backgroundImageStyle}></div>
    <div class="content">
        <div class="webcam-container">
            <Webcam bind:this={webcamComponent} on:faceCenter={handleFaceCenter} />
        </div>
        <Counter bind:this={counterComponent} on:complete={handleCounterComplete} />
        <img src={characterIconSrc} alt={$appState.userSettings.character} class="character-icon" style={characterIconStyle} />
        <div class="tip-icon tip1" style={tipsStyle}></div>
        <div class="tip-icon tip2" style={tipsStyle}></div>
        <div class="tip-icon tip3" style={tipsStyle}></div>
        <button class="start-btn" style={startBtnStyle} on:click={handleStartClick}></button>
        <div class="progress-circular-container" style={progressCircularStyle}>
            <CircularProgressBar progress={circularProgress} size={450} strokeWidth={95} />
        </div>
        <button class="stop-btn" style={stopBtnStyle} on:click={handleWomanClick}></button>
        <div class="progress-container">
            <CircleLinearProgressBar progress={progress} />
        </div>
    </div>
</div>

<style>
    /* Add component-specific styles here */
    .screen-container {
        position: relative;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }
    
    .background-video {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .background-image {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-size: cover;
        background-position: center;
        transition: background-image 0.3s ease;
        background-color: rgba(0, 0, 0, 0.3);
    }
    
    .content {
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }

    .tip-icon {
        width: 1275px; 
        height: 280px;
        border: none;
        background: transparent;
        padding: 0;
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0;
    }
    
    .tip1 {
        background-image: url('screens/camera/tip1.png');
        top: 2075px;
        animation: fadeIn 3s ease-in-out forwards;
    }
    
    .tip2 {
        background-image: url('screens/camera/tip2.png');
        top: 2305px;
        animation: fadeIn 3s ease-in-out 1s forwards;
    }
    
    .tip3 {
        background-image: url('screens/camera/tip3.png');
        top: 2527px;
        animation: fadeIn 3s ease-in-out 2s forwards;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    .character-icon {
        width: 1527px;
        height: 565px;
        border: none;
        background: transparent;
        padding: 0;
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        top: 2000px;
        transition: opacity 2s ease;
    }

    .stop-btn {
        width: 711px;
        height: 197px;
        background-image: url('screens/camera/stop.png');
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        border: none;
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        cursor: pointer;
        border-radius: 60.75px;
        transition: transform 0.1s ease-in-out;
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        top: 2790px;
    }
    
    .stop-btn:active {
        transform: translateX(-50%) scale(0.95);
    }
    
    .start-btn {
        width: 562px;
        height: 562px;
        background-image: url('screens/camera/start.png');
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        border: none;
        background-color: transparent;
        cursor: pointer;
        position: absolute;
        left: 50%;
        top: 1465px;
        transition: transform 0.3s ease-out;
    }
    
    .start-btn:active {
        transform: translateX(-50%) scale(0.95) !important;
    }

    .webcam-container {
        position: absolute;
        top:  -800px;
    }

    .progress-container {
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        top: 2995px;
    }

    .progress-circular-container {
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        top: 1512px;
    }
</style> 