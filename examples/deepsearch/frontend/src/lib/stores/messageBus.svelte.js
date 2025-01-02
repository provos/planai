import { writable } from 'svelte/store';

function createMessageBus() {
    const { subscribe, set } = writable({
        type: null,
        payload: null
    });

    return {
        subscribe,
        chatResponse: (message) => set({ type: 'chatResponse', payload: message }),
        thinkingUpdate: (message) => set({ type: 'thinkingUpdate', payload: message }),
        error: (message) => set({ type: 'error', payload: message }),
        resetMessages: () => set({ type: 'resetMessages', payload: null }),
        cleanup: () => set({ type: 'cleanup', payload: null })
    };
}

export const messageBus = createMessageBus();
