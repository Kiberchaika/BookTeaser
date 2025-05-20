<script>
    import { fade, fly } from 'svelte/transition';
    import { quintOut, quintIn } from 'svelte/easing';
    
    export let duration = 500;
    
    let isTransitioning = false;
    
    // Custom transition that combines fade and movement
    function slideAndFade(node, { duration, direction = 1 }) {
        // Use quintOut for entering (direction 1), quintIn for exiting (direction -1)
        const easing = direction > 0 ? quintIn : quintOut ;
        
        isTransitioning = true;
        
        return {
            duration,
            css: (t) => {
                const eased = easing(t);
                const translateX = direction > 0 
                    ? (1 - eased) * 25 // Start from right (100%)
                    : (eased - 1) * 25; // End to left (-100%)
                
                return `
                    opacity: ${eased};
                    transform: translateX(${translateX}%);
                `;
            },
            tick: (t) => {
                if (t === 1) {
                    isTransitioning = false;
                }
            }
        };
    }
</script>

<div class="transition-wrapper">
    {#if isTransitioning}
        <div class="overlay" />
    {/if}
    <div 
        class="content"
        in:slideAndFade={{duration: duration, direction: 1}}
        out:slideAndFade={{duration: duration, direction: -1}}
    >
        <slot></slot>
    </div>
</div>

<style>
    .transition-wrapper {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }
    
    .overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: transparent;
        z-index: 1000;
        cursor: not-allowed;
    }
    
    .content {
        position: relative;
        width: 100%;
        height: 100%;
    }
</style> 