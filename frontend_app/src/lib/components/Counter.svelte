<script>
    import { onMount, createEventDispatcher } from 'svelte';
    
    const dispatch = createEventDispatcher();
    
    let counterIndex = 0;
    let counterScale = 0;
    let counterOpacity = 0;
    let animationFrameId;
    let startTime;
    let isAnimating = false;
    
    export let webcamRef = null;
    
    const counterImages = [
        'screens/camera/5.png',
        'screens/camera/4.png',
        'screens/camera/3.png',
        'screens/camera/2.png',
        'screens/camera/1.png'
    ];
    
    function easeOutExpo(x) {
        return x === 1 ? 1 : 1 - Math.pow(2, -10 * x);
    }
    
    export function startCounter() {
        if (!isAnimating) {
            isAnimating = true;
            counterIndex = 0;
            counterScale = 0;
            counterOpacity = 0;
            startTime = null;
            animationFrameId = requestAnimationFrame(animateCounter);
        }
    }
    
    export function resetCounter() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
        }
        isAnimating = false;
        counterIndex = 0;
        counterScale = 0;
        counterOpacity = 0;
        startTime = null;
    }
    
    function animateCounter(timestamp) {
        if (!startTime) startTime = timestamp;
        const elapsed = timestamp - startTime;
        const duration = 1000; // 1 second duration
        
        if (counterIndex < counterImages.length) {
            const progress = Math.min(elapsed / duration, 1);
            counterScale = easeOutExpo(progress);
            counterOpacity = easeOutExpo(progress);
            
            if (progress >= 0.99) {
                counterIndex++;
                counterScale = 0;
                counterOpacity = 0;
                startTime = null;
                
                if (counterIndex >= counterImages.length) {
                    isAnimating = false;
                    if (webcamRef && typeof webcamRef.stopWebcam === 'function') {
                        webcamRef.stopWebcam();
                    }
                    dispatch('complete');
                    return;
                }
            }
            
            animationFrameId = requestAnimationFrame(animateCounter);
        }
    }
    
    onMount(() => {
        return () => {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
        };
    });
</script>

{#if counterIndex < counterImages.length}
    <img 
        src={counterImages[counterIndex]} 
        alt="Counter" 
        class="counter-image"
        style="transform: translateX(-50%) scale({counterScale}); opacity: {counterOpacity};"
    />
{/if}

<style>
    .counter-image {
        position: absolute;
        left: 50%;
        top: 1085px;
        width: 360px;
        height: 360px;
        object-fit: contain;
    }
</style> 