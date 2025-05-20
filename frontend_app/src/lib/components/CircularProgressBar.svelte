<script>
    export let progress = 0; // Progress value from 0 to 1
    export let size = 200; // px
    export let strokeWidth = 30; // px
    export let bgColor = "#d7c09e";
    export let progressColor = "#8b6242";
    
    // SVG calculations
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;

    $: offset = circumference * (1 - progress);
</script>

<svg
    width={size}
    height={size}
    viewBox={`0 0 ${size} ${size}`}
    class="circular-progress"
>
    <!-- Background circle -->
    <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={bgColor}
        stroke-width={strokeWidth}
    />
    <!-- Progress arc -->
    <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={progressColor}
        stroke-width={strokeWidth * 0.3}
        stroke-linecap="round"
        stroke-dasharray={circumference}
        stroke-dashoffset={offset}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
    />
</svg>

<style>
    .circular-progress {
        display: block;
        margin: auto;
    }
</style> 