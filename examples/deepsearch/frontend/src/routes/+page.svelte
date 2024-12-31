<script>
    import { onMount, onDestroy } from 'svelte';
    import { io } from 'socket.io-client';
    
    let messageInput = $state('');
    let messages = $state([]);
    let isLoading = $state(false);
    let error = $state(null);
    let socket = $state(null);

    function initializeSocket() {
        socket = io('http://localhost:5050', {
            transports: ['websocket'],
            reconnection: true
        });
        
        socket.on('connect', () => {
            console.log('Connected to chat server');
            error = null;
        });

        socket.on('connect_error', (err) => {
            console.error('Connection error:', err);
            error = 'Failed to connect to server';
        });
        
        socket.on('chat_response', (response) => {
            console.log('Received response:', response);
            isLoading = false;
            messages = [...messages, {
                role: 'assistant',
                content: response.message,
                timestamp: new Date()
            }];
        });

        socket.on('error', (err) => {
            console.error('Chat error:', err);
            isLoading = false;
            error = err;
        });
    }

    onMount(() => {
        initializeSocket();
    });

    onDestroy(() => {
        if (socket) socket.disconnect();
    });

    function handleSend() {
        if (socket && messageInput.trim()) {
            isLoading = true;
            error = null;
            
            const userMessage = {
                role: 'user',
                content: messageInput,
                timestamp: new Date()
            };
            
            messages = [...messages, userMessage];
            socket.emit('chat_message', messageInput);
            messageInput = '';
        }
    }
</script>

<main class="chat-container">
    <div class="chat-wrapper">
        <h1 class="chat-title">Chat Interface</h1>
        
        <div class="chat-box">
            <div class="messages-area">
                {#each messages as message}
                    <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
                        <div class="message-bubble {message.role === 'user' ? 'message-bubble-user' : 'message-bubble-assistant'}">
                            <p class="message-text">{message.content}</p>
                            <p class="message-timestamp">
                                {message.timestamp.toLocaleTimeString()}
                            </p>
                        </div>
                    </div>
                {/each}
                
                {#if isLoading}
                    <div class="flex justify-start">
                        <div class="message-bubble message-bubble-assistant">
                            <p class="message-text">Thinking...</p>
                        </div>
                    </div>
                {/if}
            </div>

            <div class="input-area">
                {#if error}
                    <div class="mb-2 text-sm text-red-600">{error}</div>
                {/if}
                <div class="flex gap-2">
                    <input
                        type="text"
                        bind:value={messageInput}
                        placeholder="Type your message..."
                        class="input-field"
                        onkeydown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                        disabled={isLoading}
                    />
                    <button 
                        onclick={handleSend}
                        disabled={isLoading}
                        class="send-button {isLoading ? 'send-button-disabled' : 'send-button-enabled'}"
                    >
                        {isLoading ? 'Sending...' : 'Send'}
                    </button>
                </div>
            </div>
        </div>
    </div>
</main>
