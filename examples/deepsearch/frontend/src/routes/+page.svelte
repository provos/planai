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
</script>

<main class="chat-container">
    <SessionManager />
    <ChatInterface
        {messages}
        {isLoading}
        {error}
        {thinkingUpdate}
    />
</main>
