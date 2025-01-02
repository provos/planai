<script>
    import SessionManager from '$lib/components/SessionManager.svelte';
    import ChatInterface from '$lib/components/ChatInterface.svelte';

    let messages = $state([]);
    let isLoading = $state(false);
    let error = $state(null);
    let socket = $state(null);
    let thinkingUpdate = $state('**Processing** your request...');
    let sessionId = $state(null);
    let connectionStatus = $state('disconnected');

    function handleSocketReady(event) {
        socket = event.detail;
    }

    function handleChatResponse(event) {
        console.log('Received response:', event.detail);
        isLoading = false;
        thinkingUpdate = '**Processing** your request...';
        messages = [
            ...messages,
            {
                role: 'assistant',
                content: event.detail.message,
                timestamp: new Date(),
                isMarkdown: true
            }
        ];
    }

    function handleThinkingUpdate(event) {
        console.log('Thinking update:', event.detail);
        thinkingUpdate = event.detail.message;
    }

    function handleError(event) {
        error = event.detail;
    }

    function handleResetMessages() {
        messages = [];
        isLoading = false;
    }

    function handleCleanup() {
        if (socket) {
            socket.disconnect();
        }
    }

    function handleSendMessage(message) {
        if (!sessionId || connectionStatus !== 'connected') return;
        
        isLoading = true;
        error = null;

        const userMessage = {
            role: 'user',
            content: message,
            timestamp: new Date()
        };

        messages = [...messages, userMessage];
        socket.emit('chat_message', {
            session_id: sessionId,
            message: message
        });
    }
</script>

<main class="chat-container">
    <SessionManager
        bind:sessionId
        bind:connectionStatus
        on:socketReady={handleSocketReady}
        on:chatResponse={handleChatResponse}
        on:thinkingUpdate={handleThinkingUpdate}
        on:error={handleError}
        on:resetMessages={handleResetMessages}
        on:cleanup={handleCleanup}
    />
    
    <ChatInterface
        {messages}
        {isLoading}
        {error}
        {connectionStatus}
        {thinkingUpdate}
        onSendMessage={handleSendMessage}
    />
</main>
