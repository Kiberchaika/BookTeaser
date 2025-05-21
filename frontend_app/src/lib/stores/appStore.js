import { writable } from 'svelte/store';
import { faceDetectionState, loadFaceDetectionModels } from './faceDetectionStore';

// Re-exporting face detection functionality
export { faceDetectionState, loadFaceDetectionModels };

// Application state store
export const appState = writable({
    currentScreen: 'welcome', // Current screen being displayed
    userSettings: {
        gender: null, // User's selected gender
        character: null, // Selected story scene
    },
    photo: null, // User's photo data
    faceShape: null, // Face shape 
    processingProgress: 0, // Progress of content generation
    resultUrl: null, // URL to the generated result
});

// Navigation history for back button functionality
export const navigationHistory = writable([]);

// Function to navigate to a new screen
export function navigateTo(screen) {
    appState.update(state => {
        navigationHistory.update(history => [...history, state.currentScreen]);
        return { ...state, currentScreen: screen };
    });
}

// Function to navigate back
export function navigateBack() {
    navigationHistory.update(history => {
        if (history.length > 0) {
            const previousScreen = history.pop();
            appState.update(state => ({ ...state, currentScreen: previousScreen }));
            return history;
        }
        return history;
    });
}

// WebSocket connection state
export const wsConnection = writable(null);
let wsUrl = null; // Store the WebSocket URL for reconnection

// Function to initialize WebSocket connection
export function initWebSocket(url) {
    console.log('Initializing WebSocket connection to:', url);
    wsUrl = url; // Store the URL for reconnection
    connectWebSocket();
}

// Function to establish WebSocket connection
function connectWebSocket() {
    try {
        console.log('Attempting to connect to WebSocket:', wsUrl);
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket connection established successfully');
            wsConnection.set(ws);
            
            // Send a test message to verify connection
            const testMessage = {
                type: 'test_connection',
                timestamp: new Date().toISOString()
            };
            console.log('Sending test message:', testMessage);
            ws.send(JSON.stringify(testMessage));
        };
        
        ws.onclose = (event) => {
            console.log('WebSocket connection closed:', event.code, event.reason);
            wsConnection.set(null);
            // Attempt to reconnect after 5 seconds
            setTimeout(() => {
                console.log('Attempting to reconnect...');
                connectWebSocket();
            }, 5000);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        ws.onmessage = (event) => {
            console.log('Raw WebSocket message received:', event.data);
            
            try {
                const data = JSON.parse(event.data);
                console.log('Parsed WebSocket message:', data);
                
                // Handle different message types
                if (data.status === 'success') {
                    console.log('Success message received:', data.message);
                    if (data.resultUrl) {
                        console.log('Result URL received:', data.resultUrl);
                        // Update app state with the result URL
                        appState.update(state => {
                            console.log('Updating app state with result URL:', data.resultUrl);
                            return {
                                ...state,
                                resultUrl: data.resultUrl,
                                faceShape: data.faceShape
                            };
                        });

                        // Navigate to the result screen
                        navigateTo('result');
                    }
                } else if (data.status === 'error') {
                    console.error('Error message received:', data.message);
                }
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
                console.log('Raw message that failed to parse:', event.data);
            }
        };

        // Add event listeners for debugging
        ws.addEventListener('open', () => {
            console.log('WebSocket open event fired');
        });

        ws.addEventListener('message', (event) => {
            console.log('WebSocket message event fired');
        });

        ws.addEventListener('error', (event) => {
            console.error('WebSocket error event fired:', event);
        });

        ws.addEventListener('close', (event) => {
            console.log('WebSocket close event fired:', event);
        });
        
        return ws;
    } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        // Attempt to reconnect after 5 seconds even if initialization fails
        setTimeout(() => {
            console.log('Attempting to reconnect after error...');
            connectWebSocket();
        }, 5000);
        return null;
    }
}  