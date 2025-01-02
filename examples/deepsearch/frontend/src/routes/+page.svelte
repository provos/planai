<script>
    import { writable } from 'svelte/store';
    import SessionManager from '$lib/components/SessionManager.svelte';
    import ChatInterface from '$lib/components/ChatInterface.svelte';
    import { sessionState } from '$lib/stores/sessionStore.svelte.js';

    const messages = writable([]);
    const isLoading = writable(false);
    const error = writable(null);
    const thinkingUpdate = writable('**Processing** your request...');

    $effect(() => {
        console.log('Connection status changed:', sessionState.connectionStatus);
    });

    function handleChatResponse(event) {
        console.log('Received response:', event.detail);
        isLoading.set(false);
        thinkingUpdate.set('**Processing** your request...');
        messages.update(msgs => [...msgs, {
            role: 'assistant',
            content: event.detail.message,
            timestamp: new Date(),
            isMarkdown: true
        }]);
    }

    function handleThinkingUpdate(event) {
        console.log('Thinking update:', event.detail);
        thinkingUpdate.set(event.detail.message);
    }

    function handleError(event) {
        error.set(event.detail);
    }

    function handleResetMessages() {
        messages.set([]);
        isLoading.set(false);
    }

    function handleCleanup() {
        if (sessionState.socket) {
            sessionState.socket.disconnect();
        }
    }
</script>

<main class="chat-container">
    <SessionManager
        on:chatResponse={handleChatResponse}
        on:thinkingUpdate={handleThinkingUpdate}
        on:error={handleError}
        on:resetMessages={handleResetMessages}
        on:cleanup={handleCleanup}
    />
    
    <ChatInterface
        messages={messages}
        isLoading={isLoading}
        error={error}
        thinkingUpdate={thinkingUpdate}
    />
</main>
