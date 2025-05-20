<script>
    import { onMount } from 'svelte';
    import { appState, initWebSocket, wsConnection } from '$lib/stores/appStore.js';
    import DebugPanel from '$lib/components/DebugPanel.svelte';
    import TransitionWrapper from '$lib/components/TransitionWrapper.svelte';
    import '$lib/styles/global.css';
    
    // Import preloader module
    import { preloadAssets, preloadingComplete, loadingProgress } from '$lib/utils/assetPreloader.js';

    // Import screen components
    import WelcomeScreen from '$lib/components/screens/WelcomeScreen.svelte';
    import SceneSelectionManScreen from '$lib/components/screens/SceneSelectionManScreen.svelte';
    import SceneSelectionWomanScreen from '$lib/components/screens/SceneSelectionWomanScreen.svelte';
    import SceneSelectionScreen from '$lib/components/screens/SceneSelectionManScreen.svelte';
    import CameraScreen from '$lib/components/screens/CameraScreen.svelte';
    import ProgressScreen from '$lib/components/screens/ProgressScreen.svelte';
    import ResultScreen from '$lib/components/screens/ResultScreen.svelte';

    import { loadFaceDetectionModels } from '$lib/stores/faceDetectionStore';

    // Map of screen components
    const screens = {
        'welcome': WelcomeScreen,
        'scene-selection-man': SceneSelectionManScreen,
        'scene-selection-woman': SceneSelectionWomanScreen,
        'camera': CameraScreen,
        'progress': ProgressScreen,
        'result': ResultScreen
    };

    onMount(() => {
        // Start preloading assets
        preloadAssets();

        // Load face detection models
        loadFaceDetectionModels();
        
        // Initialize WebSocket connection
        const ws = initWebSocket('ws://localhost:8765');

        return () => {
            // Clean up WebSocket connection when component is destroyed
            if (ws) {
                ws.close();
            }
        };
    });
</script>

{#if $preloadingComplete}
    <div class="app-container">
        {#each Object.entries(screens) as [key, Component]}
            {#if $appState.currentScreen === key}
                <TransitionWrapper>
                    <svelte:component this={Component} />
                </TransitionWrapper>
            {/if}
        {/each}
        <DebugPanel />
        {#if !$wsConnection}
            <div class="ws-indicator" />
        {/if}
    </div>
{:else}
    <div class="preloader">
        <div class="preloader-content">
            <h2>Loading assets...</h2>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {$loadingProgress}%"></div>
            </div>
            <p>{Math.round($loadingProgress)}%</p>
        </div>
    </div>
{/if}



<style>
    :global(html, body) {
        width: 100%;
        height: 100vh;
        margin: 0;
        padding: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #f0f0f0;
        overflow: hidden;
    }

    .app-container {
        width: 2160px;
        height: 3840px;
        position: relative;
        overflow: hidden;
        background-color: black;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        transform: scale(0.15);
        transform-origin: center center;
        min-height: 3840px;
    }
    
    .preloader {
        width: 2160px;
        height: 3840px;
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: white;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        transform: scale(0.15);
        transform-origin: center center;
        min-height: 3840px;
    }
    
    .preloader-content {
        text-align: center;
    }
    
    .progress-bar {
        width: 300px;
        height: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px auto;
    }
    
    .progress-fill {
        height: 100%;
        background-color: #4caf50;
        transition: width 0.3s ease;
    }
    
    .ws-indicator {
        position: absolute;
        bottom: 20px;
        right: 20px;
        width: 20px;
        height: 20px;
        background-color: #ff0000;
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
    }
</style>