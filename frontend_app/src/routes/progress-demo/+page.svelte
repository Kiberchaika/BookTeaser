<script>
    import { onMount } from 'svelte';
    import CircleLinearProgressBar from '$lib/components/CircleLinearProgressBar.svelte';
    import CircularProgressBar from '$lib/components/CircularProgressBar.svelte';
    
    let progress = 0;
    let autoProgress = true;
    
    onMount(() => {
        let interval;
        
        function startAutoProgress() {
            interval = setInterval(() => {
                if (autoProgress) {
                    progress += 0.005;
                    if (progress > 1) {
                        progress = 0;
                    }
                }
            }, 50);
        }
        
        startAutoProgress();
        
        return () => {
            if (interval) clearInterval(interval);
        };
    });
    
    function toggleAutoProgress() {
        autoProgress = !autoProgress;
    }
    
    function resetProgress() {
        progress = 0;
    }
</script>

<div class="container">
    <h1>Progress Bars Demo</h1>
    <p class="description">Two different styles of progress bars</p>
    
    <div class="progress-bars">
        <div class="progress-wrapper">
            <h2>Linear Progress</h2>
            <CircleLinearProgressBar {progress} />
        </div>
        
        <div class="progress-wrapper">
            <h2>Circular Progress</h2>
            <CircularProgressBar {progress} size={200} />
        </div>
    </div>
    
    <div class="controls">
        <p>Progress: {(progress * 100).toFixed(1)}%</p>
        <input type="range" min="0" max="1" step="0.01" bind:value={progress}>
        
        <div class="buttons">
            <button on:click={toggleAutoProgress}>
                {autoProgress ? 'Pause' : 'Resume'} Auto Progress
            </button>
            <button on:click={resetProgress}>Reset</button>
        </div>
    </div>
</div>

<style>
    .container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    h1 {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        text-align: center;
        margin-bottom: 1rem;
        color: #333;
    }
    
    .description {
        text-align: center;
        margin-bottom: 2rem;
        color: #666;
        font-style: italic;
    }
    
    .progress-bars {
        display: flex;
        gap: 2rem;
        justify-content: center;
        margin-bottom: 2rem;
    }
    
    .progress-wrapper {
        background-color: #333;
        padding: 2rem;
        border-radius: 8px;
        flex: 1;
        max-width: 400px;
    }
    
    .controls {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
    }
    
    input[type="range"] {
        width: 100%;
        max-width: 400px;
    }
    
    .buttons {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    button {
        padding: 0.5rem 1rem;
        background-color: #4caf50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
    }
    
    button:hover {
        background-color: #45a049;
    }
</style> 