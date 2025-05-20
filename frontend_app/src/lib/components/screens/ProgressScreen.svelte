<script>
    import CarouselLite from '$lib/components/CarouselLite.svelte';
    
    const images = [
        { id: 1, src: 'screens/progress/text1.png' },
        { id: 2, src: 'screens/progress/text2.png' },
        { id: 3, src: 'screens/progress/text3.png' }
    ];
</script>

<div class="screen-container">
    <video autoplay muted loop playsinline class="background-video">
        <source src="screens/progress/background.mp4" type="video/mp4">
    </video>
    <div class="content">
        <div class="carousel-container">
            <CarouselLite {images} autoPlay={true} autoPlayInterval={3000} />
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

    .content {
        position: relative;
        z-index: 2;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }

    .carousel-container {
        width: 1400px;
        height: 800px;
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        top: 400px;
    }
</style> 