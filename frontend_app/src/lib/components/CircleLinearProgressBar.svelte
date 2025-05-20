<script>
    export let progress = 0; // Progress value from 0 to 1
    
    // Total number of circles
    const totalCircles = 20;
    
    // Calculate circle sizes (linear increase from 8px to 40px)
    const circleSizes = Array(totalCircles).fill(0).map((_, i) => {
        return 16 + (64 * i / (totalCircles - 1));
    });
    
    // Calculate which circles should be filled based on progress
    $: filledCircles = Math.round(progress * totalCircles);
</script>

<div class="progress-container">
    {#each circleSizes as size, i}
        <div class="circle-container">
            <div 
                class="circle" 
                class:filled={i < filledCircles}
                style="width: {size}px; height: {size}px;"
            ></div>
        </div>
    {/each}
</div>

<style>
    .progress-container {
        display: flex;
        align-items: center;
        gap: 40px;
        height: 100px;
    }
    
    .circle-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .circle {
        background-color: white;
        border-radius: 50%;
        transition: background-color 0.3s ease;
    }
    
    .circle.filled {
        background-color: #d7c09e;
        box-shadow: 0 0 5px rgba(215, 192, 158, 0.7);
    }
</style> 