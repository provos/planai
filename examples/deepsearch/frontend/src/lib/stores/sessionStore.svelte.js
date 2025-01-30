import { writable } from 'svelte/store';

export const sessionState = writable({
    socket: null,
    sessionId: null,
    connectionStatus: 'disconnected'
});
