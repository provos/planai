<script>
    import { marked } from 'marked';

    export let messages = [];
    export let isLoading = false;
    export let error = null;
    export let connectionStatus = 'disconnected';
    export let thinkingUpdate = '**Processing** your request...';
    export let onSendMessage;

    let messageInput = '';

    function handleSend() {
        if (connectionStatus !== 'connected') {
            error = 'Cannot send message while disconnected';
            return;
        }
        if (messageInput.trim()) {
            onSendMessage(messageInput);
            messageInput = '';
        }
    }
</script>

<div class="chat-wrapper">
    <h1 class="chat-title">Chat Interface</h1>

    {#if connectionStatus !== 'connected'}
        <div class="connection-status {connectionStatus}">
            {connectionStatus === 'reconnecting'
                ? 'Reconnecting...'
                : connectionStatus === 'failed'
                    ? 'Connection failed'
                    : 'Disconnected'}
        </div>
    {/if}

    <div class="chat-box">
        <div class="messages-area">
            {#each messages as message}
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
                    on:keydown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                    disabled={isLoading || connectionStatus !== 'connected'}
                />
                <button
                    on:click={handleSend}
                    disabled={isLoading || connectionStatus !== 'connected'}
                    class="send-button {isLoading || connectionStatus !== 'connected'
                        ? 'send-button-disabled'
                        : 'send-button-enabled'}"
                >
                    {isLoading ? 'Sending...' : 'Send'}
                </button>
            </div>
        </div>
    </div>
</div>
