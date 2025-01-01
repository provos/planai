<script>
    import { onMount, onDestroy } from 'svelte';
    import { io } from 'socket.io-client';
    import { marked } from 'marked';
    
    let messageInput = $state('');
    let messages = $state([]);
    let isLoading = $state(false);
    let error = $state(null);
    let socket = $state(null);
    let thinkingUpdate = $state('**Processing** your request...');
    let sessionId = $state(null);
    let connectionStatus = $state('disconnected');

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
        
        // Create connection with session ID if available
        socket = io('http://localhost:5050', {
            transports: ['websocket'],
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 1000,
            query: storedSessionId ? { session_id: storedSessionId } : {}
        });

        socket.on('connect', () => {
            console.log('Connected to chat server');
            connectionStatus = 'connected';
            error = null;
        });

        socket.on('session_id', (data) => {
            const newSessionId = data.id;
            if (sessionId && sessionId !== newSessionId) {
                // Session restoration failed, clear stored session
                console.log('Session restoration failed, got new session:', newSessionId);
                messages = [];
            }
            saveSessionId(newSessionId);
            console.log('Active session ID:', newSessionId);
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            connectionStatus = 'disconnected';
            error = 'Connection lost. Attempting to reconnect...';
        });

        socket.on('reconnecting', (attemptNumber) => {
            connectionStatus = 'reconnecting';
            error = `Reconnecting... Attempt ${attemptNumber}`;
        });

        socket.on('reconnect_failed', () => {
            connectionStatus = 'failed';
            error = 'Failed to reconnect. Please refresh the page.';
        });

        socket.on('connect_error', (err) => {
            console.error('Connection error:', err);
            error = 'Failed to connect to server';
        });

        socket.on('chat_response', (response) => {
            console.log('Received response:', response);
            isLoading = false;
            thinkingUpdate = '**Processing** your request...';
            messages = [...messages, {
                role: 'assistant',
                content: response.message,
                timestamp: new Date(),
                isMarkdown: true
            }];
        });

        socket.on('thinking_update', (update) => {
            console.log('Thinking update:', update);
            thinkingUpdate = update.message;
        });

        socket.on('error', (err) => {
            console.error('Chat error:', err);
            isLoading = false;
            error = err;
            
            // Clear invalid session
            if (err === 'Invalid session ID') {
                localStorage.removeItem('chatSessionId');
                sessionId = null;
                messages = [];
            }
        });
    }

    onMount(() => {
        initializeSocket();
    });

    onDestroy(() => {
        if (socket) {
            socket.disconnect();
        }
    });

    function handleSend() {
        if (!sessionId || connectionStatus !== 'connected') {
            error = 'Cannot send message while disconnected';
            return;
        }
        if (socket && messageInput.trim()) {
            isLoading = true;
            error = null;
            
            const userMessage = {
                role: 'user',
                content: messageInput,
                timestamp: new Date()
            };
            
            messages = [...messages, userMessage];
            socket.emit('chat_message', {
                session_id: sessionId,
                message: messageInput
            });
            messageInput = '';
        }
    }
</script>

<main class="chat-container">
    <div class="chat-wrapper">
        <h1 class="chat-title">Chat Interface</h1>
        
        {#if connectionStatus !== 'connected'}
            <div class="connection-status {connectionStatus}">
                {connectionStatus === 'reconnecting' ? 'Reconnecting...' : 
                 connectionStatus === 'failed' ? 'Connection failed' :
                 'Disconnected'}
            </div>
        {/if}
        
        <div class="chat-box">
            <div class="messages-area">
                {#each messages as message}
                    <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
                        <div class="message-bubble {message.role === 'user' ? 
                            'message-bubble-user' : 
                            'message-bubble-assistant'}">
                            {#if message.isMarkdown}
                                <div class="message-text prose prose-sm dark:prose-invert">
                                    {@html marked(message.content)}
                                </div>
                            {:else}
                                <p class="message-text">{message.content}</p>
                            {/if}
                            <p class="message-timestamp">
                                {message.timestamp.toLocaleTimeString()}
                            </p>
                        </div>
                    </div>
                {/each}
                
                {#if isLoading}
                    <div class="flex justify-start">
                        <div class="message-bubble message-bubble-thinking">
                            <div class="message-text prose prose-sm dark:prose-invert thinking">
                                {@html marked(thinkingUpdate)}
                            </div>
                        </div>
                    </div>
                {/if}
            </div>

            <div class="input-area">
                {#if error}
                    <div class="error-message">{error}</div>
                {/if}
                <div class="flex gap-2">
                    <input
                        type="text"
                        bind:value={messageInput}
                        placeholder="Type your message..."
                        class="input-field"
                        onkeydown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                        disabled={isLoading || connectionStatus !== 'connected'}
                    />
                    <button 
                        onclick={handleSend}
                        disabled={isLoading || connectionStatus !== 'connected'}
                        class="send-button {(isLoading || connectionStatus !== 'connected') ? 
                            'send-button-disabled' : 'send-button-enabled'}"
                    >
                        {isLoading ? 'Sending...' : 'Send'}
                    </button>
                </div>
            </div>
        </div>
    </div>
</main>
