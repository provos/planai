<script>
	import SessionManager from '$lib/components/SessionManager.svelte';
	import ChatInterface from '$lib/components/ChatInterface.svelte';
	import { sessionState } from '$lib/stores/sessionStore.svelte.js';
	import SettingsSidebar from '$lib/components/SettingsSidebar.svelte';
	import AudioPlayer from '$lib/components/AudioPlayer.svelte';
	import { onMount } from 'svelte';

	$effect(() => {
		console.log('Connection status changed:', sessionState.connectionStatus);
	});

	onMount(() => {
		const closeTrackMenu = () => {
			const audioPlayer = document.querySelector('.audio-player');
			if (audioPlayer) {
				audioPlayer.dispatchEvent(new CustomEvent('closeMenu'));
			}
		};
		window.addEventListener('click', closeTrackMenu);
		return () => window.removeEventListener('click', closeTrackMenu);
	});
</script>

<main class="chat-container">
	<SettingsSidebar />
	<SessionManager />
	<ChatInterface />
	<div class="audio-player-wrapper">
		<AudioPlayer />
	</div>
</main>
