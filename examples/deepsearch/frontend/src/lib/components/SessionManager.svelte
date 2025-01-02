<script>
    import { onMount, onDestroy, createEventDispatcher } from 'svelte';
    import { io } from 'socket.io-client';

    const dispatch = createEventDispatcher();
    
    export let sessionId = null;
    export let connectionStatus = 'disconnected';

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
        sessionId = id;
    }

    function initializeSocket() {
        const storedSessionId = loadStoredSession();

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
            connectionStatus = 'connected';
            dispatch('error', null);
        });

        socket.on('session_id', (data) => {
            const newSessionId = data.id;
            if (sessionId && sessionId !== newSessionId) {
                console.log('Session restoration failed, got new session:', newSessionId);
                dispatch('resetMessages');
            }
            saveSessionId(newSessionId);
            console.log('Active session ID:', newSessionId);
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            connectionStatus = 'disconnected';
            dispatch('error', 'Connection lost. Attempting to reconnect...');
        });

        socket.on('reconnecting', (attemptNumber) => {
            connectionStatus = 'reconnecting';
            dispatch('error', `Reconnecting... Attempt ${attemptNumber}`);
        });

        socket.on('reconnect_failed', () => {
            connectionStatus = 'failed';
            dispatch('error', 'Failed to reconnect. Please refresh the page.');
        });

        socket.on('connect_error', (err) => {
            console.error('Connection error:', err);
            dispatch('error', 'Failed to connect to server');
        });

        socket.on('chat_response', (response) => {
            dispatch('chatResponse', response);
        });

        socket.on('thinking_update', (update) => {
            dispatch('thinkingUpdate', update);
        });

        socket.on('error', (err) => {
            console.error('Chat error:', err);
            dispatch('error', err);

            if (err === 'Invalid session ID') {
                localStorage.removeItem('chatSessionId');
                sessionId = null;
                dispatch('resetMessages');
            }
        });

        return socket;
    }

    onMount(() => {
        const socket = initializeSocket();
        dispatch('socketReady', socket);
    });

    onDestroy(() => {
        dispatch('cleanup');
    });
</script>
