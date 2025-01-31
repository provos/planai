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
<script>
	import { onMount, onDestroy } from 'svelte';
	import { io } from 'socket.io-client';
	import { sessionState } from '../stores/sessionStore.svelte.js';
	import { messageBus } from '../stores/messageBus.svelte.js';

	// Subscribe to sessionState to get socket updates
	let currentSocket = $state(null);
	sessionState.subscribe((state) => {
		currentSocket = state.socket;
	});

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
		sessionState.update((state) => ({ ...state, sessionId: id }));
	}

	$effect(() => {
		const unsubscribe = messageBus.subscribe(({ type, payload }) => {
			if (type === 'loadSettings' && currentSocket) {
				currentSocket.emit('load_settings');
			} else if (type === 'saveSettings' && currentSocket) {
				currentSocket.emit('save_settings', payload);
			} else if (type === 'validateProvider' && currentSocket) {
				currentSocket.emit('validate_provider', payload);
			} else if (type === 'listSessions' && currentSocket) {
				currentSocket.emit('list_sessions');
			} else if (type === 'retrieveSession' && currentSocket) {
				currentSocket.emit('get_session', payload);
			}
		});
		return () => unsubscribe();
	});

	function initializeSocket() {
		const storedSessionId = loadStoredSession();
		if (storedSessionId) {
			sessionState.update((state) => ({ ...state, sessionId: storedSessionId }));
		}

		const socket = io('http://localhost:5050', {
			transports: ['websocket'],
			reconnectionAttempts: Infinity,
			reconnectionDelay: 1000,
			reconnectionDelayMax: 30000,
			randomizationFactor: 0.5,
			query: storedSessionId ? { session_id: storedSessionId } : {}
		});

		socket.on('connect', () => {
			console.log('Connected to chat server');
			sessionState.update((state) => ({ ...state, connectionStatus: 'connected' }));
			messageBus.error(null);
			// Load settings on initial connection
			socket.emit('load_settings');
		});

		socket.on('session_id', (data) => {
			const newSessionId = data.id;
			if (sessionState.sessionId && sessionState.sessionId !== newSessionId) {
				console.log('Session restoration failed, got new session:', newSessionId);
				messageBus.resetMessages();
			}
			saveSessionId(newSessionId);
			console.log('Active session ID:', newSessionId);
		});

		socket.on('disconnect', () => {
			console.log('Disconnected from server');
			sessionState.update((state) => ({ ...state, connectionStatus: 'disconnected' }));
			messageBus.error('Connection lost. Attempting to reconnect...');
		});

		socket.on('reconnecting', (attemptNumber) => {
			sessionState.update((state) => ({ ...state, connectionStatus: 'reconnecting' }));
			messageBus.error(`Reconnecting... Attempt ${attemptNumber}`);
		});

		socket.on('reconnect_failed', () => {
			sessionState.update((state) => ({ ...state, connectionStatus: 'failed' }));
			messageBus.error('Failed to reconnect. Please refresh the page.');
		});

		socket.on('connect_error', (err) => {
			console.error('Connection error:', err);
			messageBus.error('Failed to connect to server');
		});

		socket.on('chat_error', (err) => {
			console.error('Chat error:', err);
			messageBus.chatError(err);
		});

		socket.on('chat_response', (response) => {
			messageBus.chatResponse(response);
		});

		socket.on('thinking_update', (update) => {
			messageBus.thinkingUpdate(update);
		});

		socket.on('settings_loaded', (settings) => {
			messageBus.settingsLoaded(settings);
		});

		socket.on('settings_saved', (status) => {
			messageBus.settingsSaved(status);
		});

		socket.on('provider_validated', (status) => {
			messageBus.providerValidated(status);
		});

		socket.on('sessions_listed', (sessions) => {
			messageBus.sessionsListed(sessions);
		});

		socket.on('session_retrieved', (session) => {
			messageBus.sessionRetrieved(session);
		});

		socket.on('error', (err) => {
			console.error('Chat error:', err);
			messageBus.error(err);

			if (err === 'Invalid session ID') {
				localStorage.removeItem('chatSessionId');
				sessionState.sessionId = null;
				messageBus.resetMessages();
			}
		});

		sessionState.update((state) => ({ ...state, socket }));
	}

	onMount(() => {
		console.log('SessionManager mounted');
		initializeSocket();
	});

	onDestroy(() => {
		messageBus.cleanup();
		if (currentSocket) {
			currentSocket.disconnect();
		}
	});
</script>
