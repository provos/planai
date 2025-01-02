<script>
    import { marked } from 'marked';
    import { sessionState } from '../stores/sessionStore.svelte.js';

    let { 
        messages,
        isLoading,
        error,
        thinkingUpdate
    } = $props();

    let messageInput = $state('');

    function handleSendMessage(message) {
        if (!sessionState.sessionId || sessionState.connectionStatus !== 'connected') return;
        
        isLoading.set(true);
        error.set(null);

        const userMessage = {
            role: 'user',
            content: message,
            timestamp: new Date()
        };

        messages.update(msgs => [...msgs, userMessage]);
        sessionState.socket?.emit('chat_message', {
            session_id: sessionState.sessionId,
            message: message
        });
    }

    $effect(() => {
        console.log('ChatInterface connection status:', sessionState.connectionStatus);
    });

    function handleSend() {
        if (sessionState.connectionStatus !== 'connected') {
            error.set('Cannot send message while disconnected');
            return;
        }
        if (messageInput.trim()) {
            handleSendMessage(messageInput);
            messageInput = '';
        }
    }

    function handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            handleSend();
        }
    }
</script>

<div class="chat-wrapper">
    <h1 class="chat-title">Chat Interface</h1>

    {#if sessionState.connectionStatus !== 'connected'}
        <div class="connection-status {sessionState.connectionStatus}">
            {sessionState.connectionStatus === 'reconnecting'
                ? 'Reconnecting...'
                : sessionState.connectionStatus === 'failed'
                    ? 'Connection failed'
                    : 'Disconnected'}
        </div>
    {/if}

    <div class="chat-box">
        <div class="messages-area">
            {#each $messages as message}
                <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
                    <div
                        class="message-bubble {message.role === 'user'
                            ? 'message-bubble-user'
                            : 'message-bubble-assistant'}"
                    >
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

            {#if $isLoading}
                <div class="flex justify-start">
                    <div class="message-bubble message-bubble-thinking">
                        <div class="message-text prose prose-sm dark:prose-invert thinking">
                            {@html marked($thinkingUpdate)}
                        </div>
                    </div>
                </div>
            {/if}
        </div>

        <div class="input-area">
            {#if $error}
                <div class="error-message">{$error}</div>
            {/if}
            <div class="flex gap-2">
                <input
                    type="text"
                    bind:value={messageInput}
                    placeholder="Type your message..."
                    class="input-field"
                    onkeydown={handleKeyDown}
                    disabled={$isLoading || sessionState.connectionStatus !== 'connected'}
                />
                <button
                    onclick={handleSend}
                    disabled={$isLoading || sessionState.connectionStatus !== 'connected'}
                    class="send-button {$isLoading || sessionState.connectionStatus !== 'connected'
                        ? 'send-button-disabled'
                        : 'send-button-enabled'}"
                >
                    {$isLoading ? 'Sending...' : 'Send'}
                </button>
            </div>
        </div>
    </div>
</div>
