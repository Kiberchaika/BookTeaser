<script>
    export let progress = 0; // Progress value from 0 to 1
    
    // Total number of circles
    const totalCircles = 20;
    
    // Calculate which circles should be growing and by how much, in reverse order
    $: circleStates = Array(totalCircles).fill(0).map((_, index) => {
        // Reverse the index to fill from right to left
        const reverseIndex = totalCircles - 1 - index;
        
        // Each circle represents 1/totalCircles of the total progress
        const circleThreshold = reverseIndex / totalCircles;
        const nextCircleThreshold = (reverseIndex + 1) / totalCircles;
        
        // If progress hasn't reached this circle yet
        if (progress < circleThreshold) {
            return 0;
        }
        
        // If progress has fully passed this circle
        if (progress >= nextCircleThreshold) {
            return 1;
        }
        
        // If progress is partially through this circle
        const circleProgress = (progress - circleThreshold) * totalCircles;
        return circleProgress;
    });
    
    // Apply smoothing effect to neighboring circles
    $: smoothedCircleStates = circleStates.map((state, index) => {
        // Define how many neighbors to affect in each direction
        const neighborEffect = 5;
        
        // Start with the circle's own state
        let finalState = state;
        
        // Apply influence from neighboring circles
        for (let i = 1; i <= neighborEffect; i++) {
            // Check neighbors ahead (to the left in visual order)
            if (index - i >= 0) {
                // Diminishing effect based on distance and neighborEffect value
                const influence = circleStates[index - i] * (1 - i/neighborEffect);
                finalState = Math.max(finalState, influence);
            }
            
            // Check neighbors behind (to the right in visual order)
            if (index + i < totalCircles) {
                // Diminishing effect based on distance and neighborEffect value
                const influence = circleStates[index + i] * (1 - i/neighborEffect);
                finalState = Math.max(finalState, influence);
            }
        }
        
        return finalState;
    });
    
    // Function to calculate circle size based on its state (0-1)
    function getCircleSize(state) {
        const minSize = 5;
        const maxSize = 30;
        return minSize + (maxSize - minSize) * state;
    }
</script>

<div class="progress-container">
    {#each smoothedCircleStates as state, i}
        <div class="circle-container">
            <div 
                class="circle" 
                style="width: {getCircleSize(state)}px; height: {getCircleSize(state)}px;"
            ></div>
        </div>
    {/each}
</div>

<style>
    .progress-container {
        display: flex;
        align-items: center;
        height: 40px;
    }
    
    .circle-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 45px; /* Fixed width for each container */
        height: 40px;
    }
    
    .circle {
        background-color: #fff;
        border-radius: 50%;
        transition: width 0.3s ease, height 0.3s ease;
        min-width: 0px;
        min-height: 0px;
        box-shadow: 0 0 5px rgba(255, 255, 255, 0.7);
    }
</style> 