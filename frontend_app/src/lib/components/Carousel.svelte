<script>
    import { onMount, tick } from 'svelte';
  
    // Props with defaults
    export let images = [
      { id: 1, src: '/api/placeholder/400/600', alt: 'Eternals', title: 'Eternals', description: 'In 5000 BC, ten superpowered Eternals are sent by the Celestial Arishem to Earth to fight the Deviants.' },
      { id: 2, src: '/api/placeholder/400/600', alt: 'Doctor Strange', title: 'Doctor Strange', description: 'A brilliant surgeon must find a new path when his hands are damaged beyond repair.' },
      { id: 3, src: '/api/placeholder/400/600', alt: 'Thor', title: 'Thor', description: 'The powerful but arrogant god Thor is cast out of Asgard to live amongst humans in Midgard (Earth).' },
      { id: 4, src: '/api/placeholder/400/600', alt: 'Guardians of the Galaxy', title: 'Guardians of the Galaxy', description: 'A group of intergalactic criminals must pull together to stop a fanatical warrior.' },
      { id: 5, src: '/api/placeholder/400/600', alt: 'Black Widow', title: 'Black Widow', description: 'Natasha Romanoff confronts the darker parts of her ledger when a dangerous conspiracy tied to her past arises.' },
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
  
    // Create a triple copy of images for infinite scrolling
    $: tripleImages = [...images, ...images, ...images];
  
    // State variables
    let carousel;
    let carouselItems = [];
    let cardHeight = 0;
    let isDragging = false;
    let isResetting = false;
    let displayIndex;
    let dragStartIndex = 0;
    let dragOffset = 0;
    let startY = 0;
  
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
      if (isResetting || displayIndex === undefined || !images.length) return;
      
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
        
        setTimeout(() => {
          displayIndex = newIndex;
          updatePositions(displayIndex, 0, true);
          
          requestAnimationFrame(() => {
            isResetting = false;
          });
        }, 50);
      }
    }
  
    // Unified drag handler
    function handleDrag(action, y) {
      if (isResetting || !images.length) return;
      
      switch (action) {
        case 'start':
          isDragging = true;
          startY = y;
          dragStartIndex = displayIndex;
          dragOffset = 0;
          carousel.classList.add('dragging');
          break;
          
        case 'move':
          if (!isDragging) return;
          const delta = y - startY;
          dragOffset = -delta / swipeThreshold;
          updatePositions(dragStartIndex, dragOffset, true);
          break;
          
        case 'end':
          if (!isDragging) return;
          
          isDragging = false;
          carousel.classList.remove('dragging');
          
          // Snap to nearest index
          displayIndex = Math.round(dragStartIndex + dragOffset);
          
          // Update logical position
          activeIndex = displayIndex % images.length;
          if (activeIndex < 0) activeIndex += images.length;
          
          dragOffset = 0;
          
          updatePositions(displayIndex, 0, false);
          
          // Check if reset needed after animation
          setTimeout(resetIfNeeded, 300);
          break;
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
  
    // Event handlers
    function handleMouseDown(e) {
      handleDrag('start', e.clientY);
      e.preventDefault();
    }
  
    function handleMouseMove(e) {
      handleDrag('move', e.clientY);
    }
  
    function handleMouseUp() {
      handleDrag('end');
    }
  
    function handleTouchStart(e) {
      handleDrag('start', e.touches[0].clientY);
    }
  
    function handleTouchMove(e) {
      handleDrag('move', e.touches[0].clientY);
      e.preventDefault();
    }
  
    function handleTouchEnd() {
      handleDrag('end');
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
      if (typeof window !== 'undefined' && images.length && displayIndex !== undefined && !isResetting && !isDragging) {
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
        return () => observer.disconnect();
      }
    });
  </script>
  
  <div class="carousel-container">
    <div class="carousel-viewport" style="perspective: {perspective}px;">
      <button
        type="button"
        class="carousel-wrapper {isDragging ? 'dragging' : ''} {isResetting ? 'resetting' : ''}"
        bind:this={carousel}
        aria-label="Image carousel"
        on:mousedown={handleMouseDown}
        on:mousemove={handleMouseMove}
        on:mouseup={handleMouseUp}
        on:mouseleave={handleMouseUp}
        on:touchstart|passive={handleTouchStart}
        on:touchmove|preventDefault={handleTouchMove}
        on:touchend={handleTouchEnd}
        on:touchcancel={handleTouchEnd}
        on:keydown={handleKeydown}
        on:transitionend={resetIfNeeded}
        tabindex="0"
      >
        {#each tripleImages as image, i (image.id + '-' + i)}
          <div class="carousel-item" data-id={image.id}>
            <div class="card">
              <img src={image.src} alt={image.alt} draggable="false" />
              <div class="card-content">
                <h2>{image.title}</h2>
                <p>{image.description}</p>
              </div>
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
      background-color: #111;
      padding: 40px 0;
      box-sizing: border-box;
    }
  
    .carousel-viewport {
      position: relative;
      width: 100%;
      max-width: 400px;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: visible;
    }
  
    .carousel-wrapper {
      position: relative;
      width: 300px;
      height: 250px;
      transform-style: preserve-3d;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: grab;
      user-select: none;
      background: transparent;
      border: none;
      padding: 0;
      outline: none;
    }
  
    .carousel-wrapper.dragging { cursor: grabbing; }
    .carousel-wrapper.resetting { pointer-events: none; }
    
    .carousel-wrapper:focus-visible {
      outline: 2px solid skyblue;
      outline-offset: 3px;
    }
  
    .carousel-item {
      position: absolute;
      width: 500px;
      height: 150px;
      transition: transform 0.5s ease-out, opacity 0.5s ease-out;
      transform-style: preserve-3d;
      backface-visibility: hidden;
    }
  
    .card {
      position: relative;
      width: 100%;
      height: 100%;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      transform-style: preserve-3d;
      background: #222;
      color: white;
    }
  
    .card img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center;
      pointer-events: none;
    }
  
    .card-content {
      position: absolute;
      bottom: 0;
      left: 0;
      width: 100%;
      padding: 20px;
      background: linear-gradient(to top, rgba(0, 0, 0, 0.8) 0%, rgba(0,0,0,0.6) 50%, transparent 100%);
      box-sizing: border-box;
    }
  
    .card-content h2 {
      margin: 0 0 10px;
      font-size: 24px;
      font-weight: bold;
    }
  
    .card-content p {
      margin: 0;
      font-size: 14px;
      opacity: 0.8;
      overflow: hidden;
      text-overflow: ellipsis;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      max-height: 3em;
      line-height: 1.4;
    }
  
    .carousel-item.active .card {
      box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6), 0 5px 15px rgba(0,0,0,0.4);
    }
  </style>