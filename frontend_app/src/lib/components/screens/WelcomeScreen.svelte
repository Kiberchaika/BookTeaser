<script>
    // Add any component logic here
    import { navigateTo, appState } from '$lib/stores/appStore.js';
    
    function handleNextClick() {
        appState.update(state => ({
            ...state,
            userSettings: {
                ...state.userSettings,
                gender: ""
            }
        }));
        navigateTo('scene-selection');
    }

</script>

<div class="screen-container">
    <video autoplay muted loop playsinline class="background-video">
        <source src="screens/welcome/background.mp4" type="video/mp4">
    </video>
    <div class="background-image"></div>
    <div class="content">
        <div class="button-container">
            <button class="next-btn" on:click={handleNextClick}></button>
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
        background-image: url('screens/welcome/text.png');
        background-size: cover;
        background-position: center;
        z-index: 1;
    }
     
    .next-btn {
        width: 1121px;
        height: 231px;
        background-image: url('screens/welcome/next.png');
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        border: none;
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        cursor: pointer;
        border-radius: 60.75px;
        transition: transform 0.1s ease-in-out;
    }
    
    .next:active {
        transform: scale(0.95);
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

    .button-container {
        display: flex;
        gap: 140px;
        margin-top: 380px;
    }
</style> 