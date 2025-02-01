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
	<div class="chat-wrapper">
		<div class="header-group">
			<h1 class="chat-title">PlanAI Research</h1>
			<div class="header-links">
				<a href="https://getplanai.com" class="brand-link">Powered by PlanAI</a>
				<span class="header-divider">•</span>
				<a href="https://www.provos.org/" class="copyright-link">© 2024 Niels Provos</a>
			</div>
		</div>
		<ChatInterface />
	</div>
	<div class="audio-player-wrapper">
		<AudioPlayer />
	</div>
</main>
