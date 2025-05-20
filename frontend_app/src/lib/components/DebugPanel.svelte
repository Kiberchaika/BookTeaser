<script>
    import { appState, navigateTo, navigateBack, wsConnection } from '../stores/appStore.js';

    // List of all available screens for debugging
    const screens = [
        'welcome',
        'scene-selection-man',
        'scene-selection-woman',
        'camera',
        'progress',
        'result'
    ];
    
    // Toggle debug panel visibility
    let isExpanded = true;
    
    function toggleExpanded() {
        isExpanded = !isExpanded;
    }
    
    // Force update app state for testing
    function updateGender(gender) {
        appState.update(state => ({
            ...state,
            userSettings: {
                ...state.userSettings,
                gender
            }
        }));
    }
    
    function setProgress(value) {
        appState.update(state => ({
            ...state,
            processingProgress: Math.min(100, Math.max(0, value))
        }));
    }
</script>

<div class="debug-panel" class:expanded={isExpanded}>
    <div class="header" on:click={toggleExpanded}>
        <h4>Debug Panel {isExpanded ? '▼' : '▶'}</h4>
    </div>
    
    {#if isExpanded}
        <div class="content">
            <div class="state-info">
                <div><strong>Current Screen:</strong> {$appState.currentScreen}</div>
                <div><strong>Gender:</strong> {$appState.userSettings.gender || 'Not selected'}</div>
                <div><strong>Character:</strong> {$appState.userSettings.character || 'Not selected'}</div>
                <div><strong>Photo:</strong> {$appState.photo ? 'Captured' : 'None'}</div>
                <div><strong>Progress:</strong> {$appState.processingProgress}%</div>
                <div><strong>WebSocket:</strong> {$wsConnection ? 'Connected' : 'Disconnected'}</div>
            </div>
            
            <div class="divider"></div>
            
            <div class="controls">
                <h5>Navigate</h5>
                <div class="screens-list">
                    {#each screens as screen}
                        <button 
                            on:click={() => navigateTo(screen)}
                            class:active={$appState.currentScreen === screen}
                        >
                            {screen}
                        </button>
                    {/each}
                </div>
                
                <button class="back-btn" on:click={navigateBack}>Back</button>
                
                <h5>Test Controls</h5>
                <div class="test-controls">
                    <div>
                        <button on:click={() => updateGender('male')}>Set Male</button>
                        <button on:click={() => updateGender('female')}>Set Female</button>
                    </div>
                    
                    <div class="progress-control">
                        <input type="range" min="0" max="100" bind:value={$appState.processingProgress} />
                        <div class="progress-buttons">
                            <button on:click={() => setProgress(0)}>0%</button>
                            <button on:click={() => setProgress(50)}>50%</button>
                            <button on:click={() => setProgress(100)}>100%</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .debug-panel {
        position: fixed;
        top: 10px;
        right: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 10px;
        border-radius: 4px;
        z-index: 9999;
        font-size: 12px;
        width: 200px;
        transition: all 0.3s ease;
        transform: scale(2);
        transform-origin: top right;
    }
    
    .debug-panel.expanded {
        width: 250px;
    }
    
    .header {
        cursor: pointer;
        user-select: none;
    }

    h4 {
        margin: 0;
        font-size: 14px;
        text-align: center;
    }
    
    h5 {
        margin: 10px 0 5px 0;
        font-size: 13px;
        text-align: center;
    }
    
    .content {
        margin-top: 10px;
    }
    
    .state-info {
        font-size: 11px;
        line-height: 1.5;
    }
    
    .divider {
        height: 1px;
        background-color: rgba(255, 255, 255, 0.3);
        margin: 10px 0;
    }

    .screens-list {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-bottom: 8px;
    }

    button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 3px;
        padding: 3px 6px;
        font-size: 11px;
        cursor: pointer;
        text-transform: capitalize;
    }
    
    button.active {
        background-color: #e74c3c;
    }

    button:hover {
        background-color: #2980b9;
    }
    
    .back-btn {
        width: 100%;
        margin-bottom: 10px;
        background-color: #9b59b6;
    }
    
    .test-controls {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .test-controls > div {
        display: flex;
        gap: 4px;
    }
    
    .progress-control {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    
    .progress-buttons {
        display: flex;
        justify-content: space-between;
    }
    
    input[type="range"] {
        width: 100%;
    }
</style> 