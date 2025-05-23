<script>
    import { onMount } from 'svelte';
    import { navigateTo } from '$lib/stores/appStore';
    import CircularProgressBar from '$lib/components/CircularProgressBar.svelte';

    let progress = 0;
    let animationFrameId;
    let startTime;

    function animate(timestamp) {
        if (!startTime) startTime = timestamp;
        const elapsed = timestamp - startTime;
        const duration = 5000; // 5 seconds

        progress = Math.min(elapsed / duration, 1);
        
        if (progress < 1) {
            animationFrameId = requestAnimationFrame(animate);
        } else {
            navigateTo('welcome');
        }
    }

    onMount(() => {
        animationFrameId = requestAnimationFrame(animate);

        return () => {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
        };
    });
</script>

<div class="blank-screen">
    <div class="progress-container">
        <CircularProgressBar {progress} size={200} strokeWidth={20} bgColor="#333333" progressColor="#ffffff" />
    </div>
</div>

<style>
    .blank-screen {
        width: 100%;
        height: 100%;
        background-color: black;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .progress-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
</style> 