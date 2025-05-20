<script>
    import { appState, navigateTo } from '$lib/stores/appStore.js';
    import QRCode from 'qrcode';
    import { onMount } from 'svelte';
    
    let qrCodeUrl = '';
    let resultVideo;
    
    onMount(async () => {
        const options = {
            color: {
                dark: '#d7c09e', // Main color
                light: '#38474510' // Transparent background
            },
            width: 578,
            margin: 0
        };
        
        qrCodeUrl = await QRCode.toDataURL('ya.ru', options);
        
        // Set video source
        resultVideo = document.querySelector('.result-video');
        if (resultVideo) {
            resultVideo.src = 'file:///C:/Users/User/Downloads/3_2.mp4';
            resultVideo.load(); // Reload the video with new source
        }
    });
    
    function handleRestartClick() {
        navigateTo('welcome');
    }
</script>

<div class="screen-container">
    <video autoplay muted loop playsinline class="background-video">
        <source src="screens/result/background.mp4" type="video/mp4">
    </video>
    <div class="background-image"></div>
    <video src="screens/result/video.mp4" autoplay muted loop playsinline class="result-video"></video>
    {#if qrCodeUrl}
        <img src={qrCodeUrl} alt="QR Code" class="qrcode" />
    {/if}
    <button class="restart-btn" on:click={handleRestartClick}></button>
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
        background-image: url('screens/result/text.png');
        background-size: cover;
        background-position: center;
        z-index: 1;
    }

    .content {
        position: relative;
        z-index: 2;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }

    .restart-btn {
        width: 711px;
        height: 197px;
        background-image: url('screens/result/restart.png');
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        border: none;
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        cursor: pointer;
        border-radius: 20px;
        transition: transform 0.1s ease-in-out;
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        top: 2650px;
        z-index: 2;
    }
    
    .restart-btn:active {
        transform: translateX(-50%) scale(0.95);
    }

    .qrcode {
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        top: 1945px;
        z-index: 2;
    }

    .result-video {
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        top: 765px;
        z-index: 2;
        border-radius: 150px;
        width: 1920px;
        height: 1080px;
        object-fit: cover;
    }
</style> 