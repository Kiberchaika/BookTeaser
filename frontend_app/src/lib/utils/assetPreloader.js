import { writable } from 'svelte/store';
import assetUrls from '/src/assets.json';

// Create stores for tracking loading progress
export const preloadingComplete = writable(false);
export const loadingProgress = writable(0);

// Preload function for images
function preloadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(url);
        img.onerror = (err) => {
            console.error(`Failed to load image: ${url}`, err);
            reject(url);
        };
        // URL already has '/static' prefix removed
        img.src = url;
    });
}

// Preload function for videos
function preloadVideo(url) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.preload = 'auto';
        
        // Set up event handlers
        video.oncanplaythrough = () => {
            // Clean up to prevent memory leaks
            video.oncanplaythrough = null;
            video.onerror = null;
            resolve(url);
        };
        
        video.onerror = (err) => {
            console.error(`Failed to load video: ${url}`, err);
            reject(url);
        };
        
        // URL already has '/static' prefix removed
        video.src = url;
        // Start loading data
        video.load();
    });
}

// Determine which preload function to use based on file extension
function getPreloadFunctionForAsset(url) {
    const extension = url.split('.').pop().toLowerCase();
    
    // List of video extensions
    const videoExtensions = ['mp4', 'webm', 'mov', 'avi', 'ogg'];
    
    if (videoExtensions.includes(extension)) {
        return preloadVideo;
    } else {
        return preloadImage;
    }
}

// Main preload function
export async function preloadAssets() {
    let loaded = 0;
    const total = assetUrls.length;
    
    if (total === 0) {
        console.warn('No assets found to preload. Check your static directory structure.');
        preloadingComplete.set(true);
        return true;
    }
    
    console.log(`Starting to preload ${total} assets...`);
    
    // Process assets in batches to prevent overwhelming the browser
    const batchSize = 5;
    const batches = [];
    
    // Split assets into batches
    for (let i = 0; i < assetUrls.length; i += batchSize) {
        batches.push(assetUrls.slice(i, i + batchSize));
    }
    
    // Process each batch sequentially
    for (const batch of batches) {
        await Promise.all(
            batch.map(url => {
                const preloadFn = getPreloadFunctionForAsset(url);
                
                return preloadFn(url)
                    .then(() => {
                        loaded++;
                        loadingProgress.set((loaded / total) * 100);
                        console.log(`Loaded asset: ${url} (${loaded}/${total})`);
                    })
                    .catch(() => {
                        // Count failed loads toward progress to avoid stalling
                        loaded++;
                        loadingProgress.set((loaded / total) * 100);
                    });
            })
        );
    }
    
    console.log('All assets preloaded!');
    preloadingComplete.set(true);
    return true;
}