<script>
    import { appState, navigateTo, navigateBack } from '$lib/stores/appStore.js';
    
    function handleCharacterSelect(character) {
        // Store the selected scene in the app state
        appState.update(state => ({
            ...state,
            userSettings: {
                ...state.userSettings,
                gender: "female",
                character: character
            }
        }));
        
        // Navigate to next screen (camera)
        navigateTo('camera');
    }

    function handleManClick() {
        navigateTo('scene-selection-man');
    }
</script>


<div class="screen-container">
    <video autoplay muted loop playsinline class="background-video">
        <source src="screens/selectionWoman/background.mp4" type="video/mp4">
    </video>
    <div class="background-image"></div>
    <div class="content">
        <div class="character-buttons">
            <button class="character-btn" on:click={() => handleCharacterSelect("olesya")}>
                <img src="screens/selectionWoman/olesya.png" alt="Олеся" />
            </button>
            <button class="character-btn" on:click={() => handleCharacterSelect("ellochka")}>
                <img src="screens/selectionWoman/ellochka.png" alt="Эллочка " />
            </button>
            <button class="character-btn" on:click={() => handleCharacterSelect("tatyana")}>
                <img src="screens/selectionWoman/tatyana.png" alt="Татьяна Ларина" />
            </button>
        </div>
    </div>
    <button class="man-btn" on:click={handleManClick}></button>
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
        background-image: url('screens/selectionWoman/text.png');
        background-size: cover;
        background-position: center;
    }
    
    .content {
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    
    .character-buttons {
        display: flex;
        flex-direction: column;
        gap: 20px;
        align-items: center;
        position: absolute;
        top: 690px;
    }
    
    .character-btn {
        width: 1527px;
        height: 565px;
        border: none;
        background: transparent;
        cursor: pointer;
        padding: 0;
        transition: transform 0.1s ease-in-out;
    }
    
    .character-btn:active {
        transform: scale(0.95);
    }
    
    .character-btn img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    .man-btn {
        width: 476px;
        height: 148px;
        background-image: url('screens/welcome/man.png');
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
        left: 310px;
        top: 2580px;
    }
    
    .man-btn:active {
        transform: scale(0.95);
    }
</style> 