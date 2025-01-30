<!--
Copyright (c) 2024 Niels Provos

This example is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

This example is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License for more details.
-->
<!--
ChatInterface Component Documentation:

This component implements a real-time chat interface with the following features:
- Handles sending and receiving messages through a WebSocket connection
- Displays messages in a chat-like interface with user/assistant bubbles
- Shows thinking/progress updates during processing
- Supports markdown rendering for assistant responses
- Manages connection state and error handling
- Provides real-time feedback through thinking updates

Key Features:
- Uses SvelteMap for reactive thinking updates
- Integrates with messageBus for event handling
- Manages WebSocket connection state
- Supports markdown rendering via marked library
- Provides visual feedback for connection status

Socket.IO Events:

Incoming Events (received):
- chatResponse: Final response from the assistant with the complete message
- thinkingUpdate: Intermediate updates about the processing state with phase and message
- error: Error messages from the server
- resetMessages: Command to clear all messages
- cleanup: Signal to disconnect the socket

Outgoing Events (sent):
- chat_message: Sends user message to server with session_id and message content
-->

<script>
	import { marked } from 'marked';
	import { SvelteMap } from 'svelte/reactivity';
	import { sessionState } from '../stores/sessionStore.svelte.js';
	import { messageBus } from '../stores/messageBus.svelte.js';
	import { FontAwesomeIcon } from '@fortawesome/svelte-fontawesome';
	import { faPaperPlane, faStop } from '@fortawesome/free-solid-svg-icons';

	// Replace props with state
	let messages = $state([]);
	let isLoading = $state(false);
	let error = $state(null);
	let thinkingUpdate = $state('**Processing** your request...');
	let messageInput = $state('');

	// Use SvelteMap instead of regular Map
	const thinkingUpdates = new SvelteMap();

	// Subscribe to message bus events
	$effect(() => {
		const unsubscribe = messageBus.subscribe(({ type, payload }) => {
			if (!type) return;

			switch (type) {
				case 'chatResponse':
					console.log('Received response:', payload);
					isLoading = false;
					// Clear all thinking updates when we get final response
					thinkingUpdates.clear();
					messages = [
						...messages,
						{
							role: 'assistant',
							content: payload.message,
							timestamp: new Date(),
							isMarkdown: true
						}
					];
					break;
				case 'chatError':
					console.error('Chat error:', payload);
					isLoading = false;
					thinkingUpdates.clear();
					messages = [
						...messages,
						{
							role: 'error',
							content: payload.message,
							timestamp: new Date()
						}
					]
					break;
				case 'thinkingUpdate':
					console.log('Thinking update:', payload);
					let phase = payload.phase;
					const message = payload.message;
					if (!phase) phase = 'unknown';
					if (phase === 'unknown' || phase === 'plan') {
						// These phases replace all other updates
						thinkingUpdates.clear();
						thinkingUpdates.set(phase, message);
					} else {
						thinkingUpdates.delete('plan');
						thinkingUpdates.delete('unknown');
						thinkingUpdates.set(phase, message);
					}
					break;
				case 'error':
					error = payload;
					break;
				case 'resetMessages':
					messages = [];
					isLoading = false;
					break;
				case 'cleanup':
					if (sessionState.socket) {
						sessionState.socket.disconnect();
					}
					break;
			}
		});

		return () => unsubscribe();
	});

	// Existing functions
	function handleSendMessage(message) {
		if (!sessionState.sessionId || sessionState.connectionStatus !== 'connected') return;

		isLoading = true;
		error = null;

		const userMessage = {
			role: 'user',
			content: message,
			timestamp: new Date()
		};

		messages = [...messages, userMessage];

		// Convert messages to simplified format for backend
		const messageHistory = messages.map(msg => ({
			role: msg.role,
			content: msg.content
		}));

		sessionState.socket?.emit('chat_message', {
			session_id: sessionState.sessionId,
			messages: messageHistory
		});
	}

	$effect(() => {
		console.log('ChatInterface connection status:', sessionState.connectionStatus);
	});

	function handleSend() {
		if (sessionState.connectionStatus !== 'connected') {
			error = 'Cannot send message while disconnected';
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

	function handleAbort() {
		console.log('Aborting current request...');
		if (sessionState.socket) {
			sessionState.socket.emit('abort', {
				session_id: sessionState.sessionId
			});
		}
		isLoading = false;
		thinkingUpdates.clear();
	}
</script>

<!-- Remove svelte:window events -->

<div class="chat-wrapper">
	<h1 class="chat-title">PlanAI Research</h1>

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
			{#each messages as message}
				<div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
					<div
						class="message-bubble {message.role === 'user'
							? 'message-bubble-user'
								: message.role === 'error'
									? 'message-bubble-error'
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
				<div class="flex flex-col justify-start gap-2">
					{#each Array.from(thinkingUpdates.entries()) as [phase, message]}
						<div class="message-bubble message-bubble-thinking">
							<div class="mb-1 text-sm text-gray-500">{phase}</div>
							<div class="message-text prose prose-sm dark:prose-invert thinking">
								{@html marked(message)}
							</div>
						</div>
					{/each}
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
					onkeydown={handleKeyDown}
					disabled={isLoading || sessionState.connectionStatus !== 'connected'}
				/>
				<button
					onclick={isLoading ? handleAbort : handleSend}
					disabled={sessionState.connectionStatus !== 'connected'}
					class="{isLoading ? 'stop-button' : 'send-button'} {!isLoading &&
					sessionState.connectionStatus !== 'connected'
						? 'send-button-disabled'
						: isLoading
							? ''
							: 'send-button-enabled'}"
					aria-label={isLoading ? 'Stop' : 'Send'}
				>
					{#if isLoading}
						<FontAwesomeIcon icon={faStop} class="send-icon" />
					{:else}
						<FontAwesomeIcon icon={faPaperPlane} class="send-icon" />
					{/if}
				</button>
			</div>
		</div>
	</div>
</div>
