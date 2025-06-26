<script>
    import { appState, navigateTo, navigateBack } from '$lib/stores/appStore.js';
    
    function handleCharacterSelect(character, gender) {
        // Store the selected scene in the app state
        appState.update(state => ({
            ...state,
            userSettings: {
                ...state.userSettings,
                gender: gender,
                character: character
            }
        }));
        
        // Navigate to next screen (camera)
        navigateTo('camera');
    }

</script>


<div class="screen-container">
    <video autoplay muted loop playsinline class="background-video">
        <source src="screens/selection/background.mp4" type="video/mp4">
    </video>
    <div class="background-image"></div>
    <div class="content">
        <div class="character-buttons">
            <button class="character-btn" on:click={() => handleCharacterSelect("assol", "female")}>
                <img src="screens/selection/assol.png" alt="Ассоль" />
            </button>
            <button class="character-btn" on:click={() => handleCharacterSelect("grey", "male")}>
                <img src="screens/selection/grey.png" alt="Грей" />
            </button>
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
        background-image: url('screens/selection/text.png');
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
        margin-top: 50px;
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

</style> 