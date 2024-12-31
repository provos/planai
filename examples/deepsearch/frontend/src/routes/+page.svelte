<script>
    import { onMount, onDestroy } from 'svelte';
    import { io } from 'socket.io-client';
    
    let messageInput = '';
    let messages = [];
    let socket;
    let isLoading = false;
    let error = null;

    onMount(() => {
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
    });

    onDestroy(() => {
        if (socket) socket.disconnect();
    });

    async function handleSend() {
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

<main class="min-h-screen bg-gray-50 flex flex-col">
    <div class="flex-1 mx-auto w-full max-w-4xl p-4">
        <h1 class="mb-8 text-center text-4xl font-bold text-gray-900">Chat Interface</h1>
        
        <div class="bg-white rounded-lg shadow-lg h-[600px] flex flex-col">
            <!-- Messages Area -->
            <div class="flex-1 p-4 overflow-y-auto space-y-4">
                {#each messages as message}
                    <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
                        <div class="max-w-[70%] {message.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100'} rounded-lg p-3">
                            <p class="text-sm">{message.content}</p>
                            <p class="text-xs mt-1 opacity-70">
                                {message.timestamp.toLocaleTimeString()}
                            </p>
                        </div>
                    </div>
                {/each}
                
                {#if isLoading}
                    <div class="flex justify-start">
                        <div class="bg-gray-100 rounded-lg p-3">
                            <p class="text-sm">Thinking...</p>
                        </div>
                    </div>
                {/if}
            </div>

            <!-- Input Area -->
            <div class="border-t p-4">
                {#if error}
                    <div class="mb-2 text-sm text-red-600">{error}</div>
                {/if}
                <div class="flex gap-2">
                    <input
                        type="text"
                        bind:value={messageInput}
                        placeholder="Type your message..."
                        class="flex-1 rounded-md border-0 px-4 py-3 text-gray-900 ring-1 ring-gray-200 focus:ring-2 focus:ring-blue-500"
                        on:keydown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                    />
                    <button 
                        on:click={handleSend}
                        class="rounded-md bg-blue-600 px-6 py-3 font-medium text-white hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                    >
                        Send
                    </button>
                </div>
            </div>
        </div>
    </div>
</main>
