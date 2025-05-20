<script>
    import { onMount, tick, onDestroy } from 'svelte';
  
    // Props with defaults
    export let images = [
      { id: 1, src: '/api/placeholder/400/600' },
      { id: 2, src: '/api/placeholder/400/600' },
      { id: 3, src: '/api/placeholder/400/600' },
      { id: 4, src: '/api/placeholder/400/600' },
      { id: 5, src: '/api/placeholder/400/600' },
    ];
    export let activeIndex = 0;
    export let visibleItems = 2;
    export let perspective = 1000;
    export let cardGap = 10;
    export let rotationAngle = 20;
    export let scaleRatio = 0.85;
    export let opacityRatio = 0.7;
    export let zOffset = 50;
    export let swipeThreshold = 100;
    export let autoPlayInterval = 5000; // 3 seconds between slides
    export let autoPlay = true;
  
    // Create a triple copy of images for infinite scrolling
    $: tripleImages = [...images, ...images, ...images];
  
    // State variables
    let carousel;
    let carouselItems = [];
    let cardHeight = 0;
    let isResetting = false;
    let displayIndex;
    let autoPlayTimer;
  
    // Auto-play functionality
    function startAutoPlay() {
      if (!autoPlay) return;
      stopAutoPlay();
      autoPlayTimer = setInterval(() => {
        navigate(1);
      }, autoPlayInterval);
    }
  
    function stopAutoPlay() {
      if (autoPlayTimer) {
        clearInterval(autoPlayTimer);
        autoPlayTimer = null;
      }
    }
  
    // Apply transforms to position items in 3D space
    function updatePositions(centerIndex, offset = 0, instant = false) {
      if (!carousel || !carouselItems.length || !images.length) return;
  
      // Get card height if not set
      if (cardHeight === 0) {
        cardHeight = carouselItems[0]?.offsetHeight || 250;
      }
  
      const effectiveCenter = centerIndex + offset;
  
      carouselItems.forEach((item, index) => {
        if (!item) return;
        
        item.style.transition = instant ? 'none' : '';
  
        const distance = index - effectiveCenter;
        const absDistance = Math.abs(distance);
  
        // Only transform visible items and immediate neighbors
        if (absDistance <= visibleItems + 0.5) {
          const z = -absDistance * zOffset;
          const y = distance * (cardHeight + cardGap);
          const rotate = -distance * rotationAngle;
          const scale = Math.pow(scaleRatio, absDistance);
          const opacity = Math.pow(opacityRatio, absDistance);
          const zIndex = tripleImages.length - Math.floor(absDistance);
  
          item.style.transform = `translateZ(${z}px) translateY(${y}px) rotateX(${rotate}deg) scale(${scale})`;
          item.style.opacity = opacity;
          item.style.zIndex = zIndex;
          item.classList.toggle('active', index === centerIndex && Math.abs(offset) < 0.5);
        } else {
          // Hide far items
          item.style.transform = `translateZ(-200px) translateY(${distance * (cardHeight + cardGap)}px) scale(0)`;
          item.style.opacity = 0;
          item.style.zIndex = 0;
        }
      });
  
      if (instant) {
        carousel.offsetHeight; // Force reflow
      }
    }
  
    // Reset to middle set of images when reaching the edge
    function resetIfNeeded() {
      if (isResetting || displayIndex === undefined || !images.length || !carousel) return;
      
      const sectionSize = images.length;
      let needsReset = false;
      let newIndex = displayIndex;
  
      // Check if we need to reset position
      if (displayIndex < sectionSize) {
        newIndex = displayIndex + sectionSize;
        needsReset = true;
      } else if (displayIndex >= 2 * sectionSize) {
        newIndex = displayIndex - sectionSize;
        needsReset = true;
      }
  
      if (needsReset) {
        isResetting = true;
        
        // Add transition class before reset
        if (carousel) {
          carousel.classList.add('resetting');
        }
        
        // Wait for the current transition to complete before resetting
        setTimeout(() => {
          displayIndex = newIndex;
          updatePositions(displayIndex, 0, true);
          
          // Remove transition class after reset with a longer delay
          requestAnimationFrame(() => {
            setTimeout(() => {
              if (carousel) {
                carousel.classList.remove('resetting');
              }
              isResetting = false;
            }, 50);
          });
        }, 1200); // Match the transition duration from CSS
      }
    }
  
    // Navigation helpers
    function navigate(direction) {
      if (isResetting || displayIndex === undefined || !images.length) return;
      
      displayIndex += direction;
      activeIndex = displayIndex % images.length;
      if (activeIndex < 0) activeIndex += images.length;
      
      updatePositions(displayIndex, 0, false);
      setTimeout(resetIfNeeded, 300);
    }
  
    function handleKeydown(e) {
      if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
        navigate(-1);
        e.preventDefault();
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
        navigate(1);
        e.preventDefault();
      }
    }
  
    // Handle external activeIndex changes
    $: {
      if (typeof window !== 'undefined' && images.length && displayIndex !== undefined && !isResetting) {
        // Ensure activeIndex is within bounds
        const validIndex = Math.max(0, Math.min(activeIndex, images.length - 1));
        
        if (validIndex !== activeIndex) {
          activeIndex = validIndex;
        } else {
          // Calculate current logical position
          const currentActive = displayIndex % images.length;
          const normalizedActive = currentActive < 0 ? currentActive + images.length : currentActive;
          
          if (activeIndex !== normalizedActive) {
            // Position at the middle section
            displayIndex = images.length + activeIndex;
            tick().then(() => updatePositions(displayIndex, 0, false));
          }
        }
      }
    }
  
    // Initialize carousel
    onMount(async () => {
      if (!carousel) return;
      
      await tick();
      carouselItems = Array.from(carousel.querySelectorAll('.carousel-item'));
      
      if (images.length > 0) {
        activeIndex = Math.max(0, Math.min(activeIndex, images.length - 1));
        displayIndex = images.length + activeIndex; // Start in middle clone
        cardHeight = carouselItems[0]?.offsetHeight || 250;
        
        updatePositions(displayIndex, 0, true);
        startAutoPlay();
        
        // Set up resize observer
        const observer = new ResizeObserver(() => {
          if (carouselItems.length > 0) {
            const newHeight = carouselItems[0].offsetHeight;
            if (newHeight > 0 && newHeight !== cardHeight) {
              cardHeight = newHeight;
              updatePositions(displayIndex, 0, true);
            }
          }
        });
        
        observer.observe(carousel);
        return () => {
          observer.disconnect();
          stopAutoPlay();
        };
      }
    });
  
    // Cleanup on component destruction
    onDestroy(() => {
      stopAutoPlay();
    });
  </script>
  
  <div class="carousel-container">
    <div class="carousel-viewport" style="perspective: {perspective}px;">
      <button
        type="button"
        class="carousel-wrapper {isResetting ? 'resetting' : ''}"
        bind:this={carousel}
        aria-label="Image carousel"
        on:keydown={handleKeydown}
        on:transitionend={resetIfNeeded}
        tabindex="0"
      >
        {#each tripleImages as image, i (image.id + '-' + i)}
          <div class="carousel-item" data-id={image.id}>
            <div class="card">
              <img src={image.src} draggable="false" />
            </div>
          </div>
        {/each}
      </button>
    </div>
  </div>
  
  <style>
    .carousel-container {
      position: relative;
      width: 100%;
      height: 100%;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 40px 0;
      box-sizing: border-box;
      background: transparent;
    }
  
    .carousel-viewport {
      position: relative;
      width: 100%;
      max-width: 1200px;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: visible;
      background: transparent;
    }
  
    .carousel-wrapper {
      position: relative;
      width: 1200px;
      height: 250px;
      transform-style: preserve-3d;
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
      background: transparent;
      border: none;
      padding: 0;
      outline: none;
      box-shadow: none;
      transition: transform 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
  
    .carousel-wrapper.resetting { 
      pointer-events: none;
      transition: none;
    }
    
    .carousel-item {
      position: absolute;
      width: 945px;
      height: 292px;
      transition: transform 1.2s cubic-bezier(0.4, 0, 0.2, 1), opacity 1.2s cubic-bezier(0.4, 0, 0.2, 1);
      transform-style: preserve-3d;
      backface-visibility: hidden;
      will-change: transform, opacity;
    }
  
    .card {
      position: relative;
      width: 100%;
      height: 100%;
      overflow: hidden;
      transform-style: preserve-3d;
    }
  
    .card img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center;
      pointer-events: none;
    }
  
</style>