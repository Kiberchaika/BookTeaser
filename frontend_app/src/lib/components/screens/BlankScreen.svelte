<script>
    import { onMount } from 'svelte';
    import { navigateTo } from '$lib/stores/appStore';

    onMount(() => {
        const timer = setTimeout(() => {
            navigateTo('welcome');
        }, 5000);

        return () => clearTimeout(timer);
    });
</script>

<div class="blank-screen">
    <!-- Empty screen -->
</div>

<style>
    .blank-screen {
        width: 100%;
        height: 100%;
        background-color: black;
    }
</style> 