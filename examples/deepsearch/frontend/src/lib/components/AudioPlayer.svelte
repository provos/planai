<script>
	import { currentTrack, tracks } from '$lib/stores/trackStore';

	let time = $state(0);
	let duration = $state(0);
	let paused = $state(true);
	let showTrackMenu = $state(false);

	function format(time) {
		if (isNaN(time)) return '...';
		const minutes = Math.floor(time / 60);
		const seconds = Math.floor(time % 60);
		return `${minutes}:${seconds < 10 ? `0${seconds}` : seconds}`;
	}

	function selectTrack(track) {
		$currentTrack = track;
		time = 0;
		paused = false;
		showTrackMenu = false;
	}

	function handleKeydown(event) {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			event.currentTarget.click();
		}
		if (event.key === 'Escape' && showTrackMenu) {
			showTrackMenu = false;
		}
	}
</script>

<div class="audio-player">
	<audio
		src={$currentTrack.src}
		bind:currentTime={time}
		bind:duration
		bind:paused
		onended={() => {
			time = 0;
		}}
	></audio>

	<div class="player-controls">
		<button
			class="player-button"
			aria-label={paused ? 'play' : 'pause'}
			onclick={() => (paused = !paused)}
		>
			{#if paused}
				<svg
					xmlns="http://www.w3.org/2000/svg"
					class="h-5 w-5"
					viewBox="0 0 24 24"
					fill="currentColor"
				>
					<path d="M8 5v14l11-7z" />
				</svg>
			{:else}
				<svg
					xmlns="http://www.w3.org/2000/svg"
					class="h-5 w-5"
					viewBox="0 0 24 24"
					fill="currentColor"
				>
					<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
				</svg>
			{/if}
		</button>

		<div class="track-info-wrapper">
			<button
				class="track-info-button"
				onclick={() => (showTrackMenu = !showTrackMenu)}
				onkeydown={handleKeydown}
				aria-haspopup="listbox"
				aria-controls="track-menu"
				aria-expanded={showTrackMenu}
				role="combobox"
			>
				<div class="track-title">{$currentTrack.title}</div>
				<div class="track-artist">{$currentTrack.artist}</div>
				<svg
					class="track-selector-icon"
					xmlns="http://www.w3.org/2000/svg"
					viewBox="0 0 20 20"
					fill="currentColor"
				>
					<path
						d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
					/>
				</svg>
			</button>

			{#if showTrackMenu}
				<div
					class="track-menu"
					onclick={(e) => e.stopPropagation()}
					onkeydown={handleKeydown}
					role="listbox"
					tabindex="-1"
				>
					{#each tracks as track}
						<button
							class="track-menu-item"
							class:active={track == $currentTrack}
							onclick={() => selectTrack(track)}
							onkeydown={handleKeydown}
							role="option"
							aria-selected={track == $currentTrack}
						>
							<div class="track-title">{track.title}</div>
							<div class="track-artist">{track.artist}</div>
						</button>
					{/each}
				</div>
			{/if}
		</div>

		<div class="player-timeline">
			<span class="time-display">{format(time)}</span>
			<div
				class="timeline-slider"
				onpointerdown={(e) => {
					const div = e.currentTarget;
					function seek(e) {
						const { left, width } = div.getBoundingClientRect();
						let p = (e.clientX - left) / width;
						if (p < 0) p = 0;
						if (p > 1) p = 1;
						time = p * duration;
					}
					seek(e);
					window.addEventListener('pointermove', seek);
					window.addEventListener(
						'pointerup',
						() => {
							window.removeEventListener('pointermove', seek);
						},
						{ once: true }
					);
				}}
			>
				<div class="timeline-progress" style="width: {(time / duration) * 100}%"></div>
			</div>
			<span class="time-display">{duration ? format(duration) : '--:--'}</span>
		</div>
	</div>
</div>
