<script>
    import SessionManager from '$lib/components/SessionManager.svelte';
    import ChatInterface from '$lib/components/ChatInterface.svelte';
    import { sessionState } from '$lib/stores/sessionStore.svelte.js';
    import SettingsSidebar from '$lib/components/SettingsSidebar.svelte';

    $effect(() => {
        console.log('Connection status changed:', sessionState.connectionStatus);
    });
</script>

<main class="chat-container">
    <SettingsSidebar />
    <SessionManager />
    <ChatInterface />
</main>
