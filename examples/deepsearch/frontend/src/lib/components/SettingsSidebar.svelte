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
	import { FontAwesomeIcon } from '@fortawesome/svelte-fontawesome';
	import { faPaperPlane, faStop, faBars } from '@fortawesome/free-solid-svg-icons';
	import { messageBus } from '../stores/messageBus.svelte.js';
	import { configState } from '../stores/configStore.svelte.js';
	import { sessionState } from '../stores/sessionStore.svelte.js';

	let showSettings = $state(false);
	let showSessionHistory = $state(false);
	let providers = $state({
		ollama: { available: false, models: [] },
		openai: { available: false, models: [], hasKey: false },
		anthropic: { available: false, models: [], hasKey: false }
	});

	let config = $state({
		serperApiKey: '',
		openAiApiKey: '',
		anthropicApiKey: '',
		provider: '',
		modelName: '',
		ollamaHost: 'localhost:11434' // Add default Ollama host
	});

	let hasSerperKey = $state(false);
	let sessions = $state([]);

	let availableProviders = $derived(
		Object.entries(providers)
			.filter(([_, info]) => info.available)
			.map(([key, _]) => key)
	);

	let availableModels = $derived(config.provider ? providers[config.provider]?.models || [] : []);

	let canSave = $derived(
		// Only require serperApiKey if we don't already have one stored
		(hasSerperKey || config.serperApiKey) &&
			config.provider &&
			config.modelName &&
			availableProviders.includes(config.provider)
	);

	let settingsLoaded = false;

	// Add debounce helper
	function debounce(func, wait) {
		let timeout;
		return function executedFunction(...args) {
			const later = () => {
				clearTimeout(timeout);
				func(...args);
			};
			clearTimeout(timeout);
			timeout = setTimeout(later, wait);
		};
	}

	// Debounced validation functions
	const debouncedValidateProvider = debounce((provider, apiKey) => {
		if (apiKey) {
			console.log('Validating provider:', provider, provider === 'ollama' ? apiKey : '********');
			messageBus.validateProvider(provider, apiKey);
		} else {
			providers[provider] = {
				...providers[provider],
				available: false,
				models: [],
				hasKey: false
			};
		}
	}, 500); // 500ms delay

	let currentConnectionStatus;

	// Subscribe to the store changes
	sessionState.subscribe((state) => {
		currentConnectionStatus = state.connectionStatus;
	});

	$effect(() => {
		// Use the local variable instead of store directly
		if (currentConnectionStatus === 'connected' && !settingsLoaded) {
			console.log('Loading settings on initial connection');
			loadSettings();
			settingsLoaded = true;
		}
	});

	// Reset the settingsLoaded flag if we disconnect
	$effect(() => {
		if (currentConnectionStatus === 'disconnected') {
			settingsLoaded = false;
		}
	});

	$effect(() => {
		const unsubscribe = messageBus.subscribe(({ type, payload }) => {
			if (type === 'settingsLoaded') {
				console.log('Settings loaded:', payload);

				// First update the providers state
				providers = payload.providers;

				// Then update the config state atomically
				const updates = {
					serperApiKey: '', // Don't load actual key value
					openAiApiKey: '', // Don't load actual key value
					anthropicApiKey: '', // Don't load actual key value
					ollamaHost: payload.ollamaHost || 'localhost:11434',
					provider: '', // Start with empty provider
					modelName: '' // Start with empty model
				};

				hasSerperKey = payload.serperApiKey;

				// Verify the saved provider is actually available
				const savedProvider = payload.provider;
				const savedModel = payload.modelName;

				const isProviderAvailable =
					savedProvider &&
					payload.providers[savedProvider]?.available &&
					payload.providers[savedProvider]?.models.includes(savedModel);

				if (isProviderAvailable) {
					console.log('Restoring valid saved provider/model:', savedProvider, savedModel);
					updates.provider = savedProvider;
					updates.modelName = savedModel;
				} else {
					// Find first available provider
					const firstProvider = Object.entries(payload.providers).find(
						([_, info]) => info.available
					)?.[0];

					if (firstProvider) {
						console.log('Auto-selecting first available provider:', firstProvider);
						updates.provider = firstProvider;
						const models = payload.providers[firstProvider].models;
						if (models && models.length > 0) {
							console.log('Auto-selecting first model:', models[0]);
							updates.modelName = models[0];
						}
					}
				}

				// Update config state all at once
				config = { ...config, ...updates };
			} else if (type === 'settingsSaved') {
				console.log('Settings saved:', payload);
				showSettings = false;
			} else if (type === 'providerValidated') {
				console.log('Provider validated:', payload);
				const { provider, isValid, availableModels } = payload;
				providers[provider] = {
					...providers[provider],
					available: isValid,
					models: availableModels,
					hasKey: true
				};
				console.log('Updated providers:', providers);
				console.log('Available providers after update:', availableProviders);

				// If this is the current provider and it's no longer valid, reset model
				if (config.provider === provider && !isValid) {
					console.log('Resetting model due to invalid provider');
					config.modelName = '';
				}
			} else if (type === 'sessionsListed') {
				// Fix: ensure payload has the correct structure
				console.log('Received sessions:', payload);
				if (payload && payload.sessions) {
					sessions = payload.sessions;
				} else {
					console.error('Invalid sessions payload:', payload);
					sessions = [];
				}
			}
		});
		return () => unsubscribe();
	});

	$effect(() => {
		const snapshot = $state.snapshot(config);
		const providerSnapshot = $state.snapshot(providers);
		const isValid =
			(hasSerperKey || snapshot.serperApiKey) &&
			snapshot.provider &&
			snapshot.modelName &&
			providerSnapshot[snapshot.provider]?.available;

		configState.update((state) => ({
			...state,
			isValid,
			provider: snapshot.provider,
			modelName: snapshot.modelName
		}));

		console.log('Config state updated:', {
			isValid,
			provider: snapshot.provider,
			modelName: snapshot.modelName
		});
	});

	async function loadSettings() {
		messageBus.loadSettings();
	}

	async function saveSettings() {
		console.log('Saving settings:', config.provider, config.modelName);
		messageBus.saveSettings(config);
	}

	async function loadSessions() {
		console.log('Loading sessions');
		messageBus.listSessions();
	}

	async function loadSession(sessionId) {
		console.log('Loading session:', sessionId);
		messageBus.getSession(sessionId);
		showSessionHistory = false;
	}

	function onProviderChange(event) {
		const provider = event.target.value;
		config.provider = provider;
		config.modelName = ''; // Reset model when provider changes
	}

	function onApiKeyChange(provider, apiKey) {
		debouncedValidateProvider(provider, apiKey);
	}

	function onOllamaHostChange(event) {
		if (config.ollamaHost) {
			debouncedValidateProvider('ollama', config.ollamaHost);
		}
	}

	// Helper to determine if key is invalid (has content but provider not available)
	let hasInvalidKey = $derived({
		openai: config.openAiApiKey && !providers.openai.available,
		anthropic: config.anthropicApiKey && !providers.anthropic.available
	});
</script>

<div class="settings-sidebar">
	<button
		class="settings-button"
		onclick={() => {
			showSessionHistory = true;
			loadSessions();
		}}
		aria-label="Session History"
	>
		<FontAwesomeIcon icon={faBars} />
	</button>

	<button
		class="settings-button"
		onclick={() => {
			showSettings = true;
			loadSettings();
		}}
		aria-label="Open Settings"
	>
		<span class="text-xl">&#9881;</span>
	</button>
</div>

{#if showSessionHistory}
	<dialog class="settings-dialog" aria-labelledby="sessions-title" open>
		<button class="settings-overlay" onclick={() => (showSessionHistory = false)}>
			<span class="sr-only">Close sessions</span>
		</button>

		<section class="settings-panel" role="document">
			<div class="sessions-list">
				{#if sessions.length === 0}
					<div class="session-empty">No previous sessions found</div>
				{:else}
					{#each sessions as session}
						<button class="session-item" onclick={() => loadSession(session.id)}>
							<p class="session-message">{session.first_message}</p>
						</button>
					{/each}
				{/if}
			</div>
		</section>
	</dialog>
{/if}

{#if showSettings}
	<dialog class="settings-dialog" aria-labelledby="settings-title" open>
		<button
			class="settings-overlay"
			onclick={() => (showSettings = false)}
			onkeydown={(e) => {
				if (e.key === 'Escape') showSettings = false;
			}}
		>
			<span class="sr-only">Close settings</span>
		</button>

		<section class="settings-panel" role="document">
			<h2 id="settings-title" class="settings-title">Settings</h2>
			<form
				onsubmit={(e) => {
					e.preventDefault();
					saveSettings();
				}}
				class="settings-form"
			>
				<!-- Form groups -->
				<div class="form-group">
					<label for="serper-key" class="form-label">
						Serper API Key {#if !hasSerperKey}<span class="text-red-500">*</span>{/if}
					</label>
					<input
						id="serper-key"
						type="password"
						bind:value={config.serperApiKey}
						placeholder={hasSerperKey ? 'Key stored' : 'Enter your Serper key'}
						class="form-input"
						required={!hasSerperKey}
					/>
				</div>

				<div class="form-group">
					<label for="openai-key" class="form-label"> OpenAI API Key </label>
					<input
						id="openai-key"
						type="password"
						bind:value={config.openAiApiKey}
						oninput={(e) => onApiKeyChange('openai', e.target.value)}
						placeholder={providers.openai?.hasKey ? 'Key stored' : 'Enter your OpenAI key'}
						class="form-input"
						class:border-red-500={hasInvalidKey.openai}
						class:border-gray-300={!hasInvalidKey.openai}
						class:dark:border-red-500={hasInvalidKey.openai}
						class:dark:border-gray-600={!hasInvalidKey.openai}
					/>
					{#if hasInvalidKey.openai}
						<p class="mt-1 text-sm text-red-500">Invalid API key</p>
					{/if}
				</div>

				<div class="form-group">
					<label for="anthropic-key" class="form-label"> Anthropic API Key </label>
					<input
						id="anthropic-key"
						type="password"
						bind:value={config.anthropicApiKey}
						oninput={(e) => onApiKeyChange('anthropic', e.target.value)}
						placeholder={providers.anthropic?.hasKey ? 'Key stored' : 'Enter your Anthropic key'}
						class="form-input"
						class:border-red-500={hasInvalidKey.anthropic}
						class:border-gray-300={!hasInvalidKey.anthropic}
						class:dark:border-red-500={hasInvalidKey.anthropic}
						class:dark:border-gray-600={!hasInvalidKey.anthropic}
					/>
					{#if hasInvalidKey.anthropic}
						<p class="mt-1 text-sm text-red-500">Invalid API key</p>
					{/if}
				</div>

				<div class="form-group">
					<label for="ollama-host" class="form-label"> Ollama Host </label>
					<input
						id="ollama-host"
						type="text"
						bind:value={config.ollamaHost}
						oninput={onOllamaHostChange}
						placeholder="localhost:11434"
						class="form-input"
					/>
				</div>

				<div class="form-group">
					<label for="provider-select" class="form-label"> Provider </label>
					<select
						id="provider-select"
						bind:value={config.provider}
						onchange={onProviderChange}
						class="form-input"
					>
						<option value="">Select Provider</option>
						{#each availableProviders as provider}
							<option value={provider}>
								{provider.charAt(0).toUpperCase() + provider.slice(1)}
							</option>
						{/each}
					</select>
				</div>

				{#if config.provider && providers[config.provider]?.models.length > 0}
					<div class="form-group">
						<label for="model-name" class="form-label"> Model Name </label>
						<select id="model-name" bind:value={config.modelName} class="form-input">
							<option value="">Select Model</option>
							{#each availableModels as model}
								<option value={model}>{model}</option>
							{/each}
						</select>
					</div>
				{/if}

				<div class="form-actions">
					<button type="button" class="button-cancel" onclick={() => (showSettings = false)}>
						Cancel
					</button>
					<button
						type="submit"
						class="button-save"
						class:button-save-enabled={canSave}
						class:button-save-disabled={!canSave}
						disabled={!canSave}
					>
						Save
					</button>
				</div>
			</form>
		</section>
	</dialog>
{/if}
