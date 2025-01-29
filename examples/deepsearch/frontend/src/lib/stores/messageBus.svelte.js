import { writable } from 'svelte/store';

function createMessageBus() {
    const { subscribe, set } = writable({
        type: null,
        payload: null
    });

    return {
        subscribe,
        chatResponse: (message) => set({ type: 'chatResponse', payload: message }),
        chatError: (message) => set({ type: 'chatError', payload: message }),
        thinkingUpdate: (message) => set({ type: 'thinkingUpdate', payload: message }),
        error: (message) => set({ type: 'error', payload: message }),
        resetMessages: () => set({ type: 'resetMessages', payload: null }),
        cleanup: () => set({ type: 'cleanup', payload: null }),
        settingsLoaded: (settings) => set({ type: 'settingsLoaded', payload: settings }),
        settingsSaved: (status) => set({ type: 'settingsSaved', payload: status }),
        loadSettings: () => set({ type: 'loadSettings', payload: null }),
        saveSettings: (settings) => set({ type: 'saveSettings', payload: settings }),
        validateProvider: (provider, apiKey) => set({ type: 'validateProvider', payload: { provider, apiKey } }),
        providerValidated: (result) => set({ type: 'providerValidated', payload: result })
    };
}

export const messageBus = createMessageBus();
