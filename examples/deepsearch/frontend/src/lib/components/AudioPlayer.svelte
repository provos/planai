<!--
Copyright (c) 2025 Niels Provos

This example is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

This example is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License for more details.
-->
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

		<div class="track-info-panel">
			<p class="track-info-text hidden md:block">{$currentTrack.info}</p>
			<a
				href={$currentTrack.link}
				target="_blank"
				rel="noopener noreferrer"
				class="track-info-link"
			>
				{#if !$currentTrack.info}
					Learn more
				{:else}
					<span class="hidden md:inline">Learn more</span>
					<span class="md:hidden">More info</span>
				{/if}
				<svg
					xmlns="http://www.w3.org/2000/svg"
					class="ml-1 inline-block h-4 w-4"
					viewBox="0 0 20 20"
					fill="currentColor"
				>
					<path
						d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z"
					/>
					<path
						d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z"
					/>
				</svg>
			</a>
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
