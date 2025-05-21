<script>
    import { appState, navigateTo, navigateBack } from '$lib/stores/appStore.js';
    
    function handleCharacterSelect(character) {
        // Store the selected character in the app state
        appState.update(state => ({
            ...state,
            userSettings: {
                ...state.userSettings,
                gender: "male",
                character: character
            }
        }));
        
        // Navigate to next screen (camera)
        navigateTo('camera');
    }

    function handleWomanClick() {
        navigateTo('scene-selection-woman');
    }
</script>

 
<div class="screen-container">
    <video autoplay muted loop playsinline class="background-video">
        <source src="screens/selectionMan/background.mp4" type="video/mp4">
    </video>
    <div class="background-image"></div>
    <div class="content">
        <div class="character-buttons">
            <button class="character-btn" on:click={() => handleCharacterSelect("dartanyan")}>
                <img src="screens/selectionMan/dartanyan.png" alt="Д'Артаньян" />
            </button>
            <button class="character-btn" on:click={() => handleCharacterSelect("don_kihot")}>
                <img src="screens/selectionMan/don_kihot.png" alt="Дон Кихот" />
            </button>
            <button class="character-btn" on:click={() => handleCharacterSelect("ostap_bender")}>
                <img src="screens/selectionMan/ostap_bender.png" alt="Остап Бендер" />
            </button>
        </div>
    </div>
    <button class="woman-btn" on:click={handleWomanClick}></button>
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
        background-image: url('screens/selectionMan/text.png');
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

    .woman-btn {
        width: 476px;
        height: 148px;
        background-image: url('screens/welcome/woman.png');
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        border: none;
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        cursor: pointer;
        border-radius: 45.56px;
        transition: transform 0.1s ease-in-out;
        position: absolute;
        left: 310px;
        top: 2580px;
    }
    
    .woman-btn:active {
        transform: scale(0.95);
    }
</style> 