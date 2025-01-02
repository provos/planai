<script>
    import { onMount, onDestroy } from 'svelte';
    import { io } from 'socket.io-client';
    import { sessionState } from '../stores/sessionStore.svelte.js';
    import { messageBus } from '../stores/messageBus.svelte.js';

    function loadStoredSession() {
        const storedId = localStorage.getItem('chatSessionId');
        if (storedId) {
            console.log('Restored session ID from storage:', storedId);
            return storedId;
        }
        return null;
    }

    function saveSessionId(id) {
        localStorage.setItem('chatSessionId', id);
        sessionState.sessionId = id;
    }

    function initializeSocket() {
        const storedSessionId = loadStoredSession();
        if (storedSessionId) {
            sessionState.sessionId = storedSessionId;
        }

        const socket = io('http://localhost:5050', {
            transports: ['websocket'],
            reconnectionAttempts: Infinity,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 30000,
            randomizationFactor: 0.5,
            query: storedSessionId ? { session_id: storedSessionId } : {}
        });

        socket.on('connect', () => {
            console.log('Connected to chat server');
            sessionState.connectionStatus = 'connected';
            messageBus.error(null);
        });

        socket.on('session_id', (data) => {
            const newSessionId = data.id;
            if (sessionState.sessionId && sessionState.sessionId !== newSessionId) {
                console.log('Session restoration failed, got new session:', newSessionId);
                messageBus.resetMessages();
            }
            saveSessionId(newSessionId);
            console.log('Active session ID:', newSessionId);
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            sessionState.connectionStatus = 'disconnected';
            messageBus.error('Connection lost. Attempting to reconnect...');
        });

        socket.on('reconnecting', (attemptNumber) => {
            sessionState.connectionStatus = 'reconnecting';
            messageBus.error(`Reconnecting... Attempt ${attemptNumber}`);
        });

        socket.on('reconnect_failed', () => {
            sessionState.connectionStatus = 'failed';
            messageBus.error('Failed to reconnect. Please refresh the page.');
        });

        socket.on('connect_error', (err) => {
            console.error('Connection error:', err);
            messageBus.error('Failed to connect to server');
        });

        socket.on('chat_response', (response) => {
            messageBus.chatResponse(response);
        });

        socket.on('thinking_update', (update) => {
            messageBus.thinkingUpdate(update);
        });

        socket.on('error', (err) => {
            console.error('Chat error:', err);
            messageBus.error(err);

            if (err === 'Invalid session ID') {
                localStorage.removeItem('chatSessionId');
                sessionState.sessionId = null;
                messageBus.resetMessages();
            }
        });

        sessionState.socket = socket;
    }

    onMount(() => {
        console.log('SessionManager mounted');
        initializeSocket();
    });

    onDestroy(() => {
        messageBus.cleanup();
        if (sessionState.socket) {
            sessionState.socket.disconnect();
        }
    });
</script>
