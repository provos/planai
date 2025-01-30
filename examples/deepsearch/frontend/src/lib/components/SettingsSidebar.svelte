<script>
	import { FontAwesomeIcon } from '@fortawesome/svelte-fontawesome';
	import { faPaperPlane, faStop } from '@fortawesome/free-solid-svg-icons';
	import { messageBus } from '../stores/messageBus.svelte.js';

	let showSettings = $state(false);
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

	$effect(() => {
		const unsubscribe = messageBus.subscribe(({ type, payload }) => {
			if (type === 'settingsLoaded') {
				console.log('Settings loaded:', payload);
				config.serperApiKey = ''; // Don't load actual key value
				config.openAiApiKey = ''; // Don't load actual key value
				config.anthropicApiKey = ''; // Don't load actual key value
				config.provider = payload.provider || '';
				config.modelName = payload.modelName || '';
				config.ollamaHost = payload.ollamaHost || 'localhost:11434'; // Load Ollama host
				hasSerperKey = payload.serperApiKey; // Boolean indicating if key exists
				providers = payload.providers;

				// Wait for providers to be set and availableProviders to be computed
				const savedProvider = payload.provider;
				const savedModel = payload.modelName;

				// Check if the saved provider is actually available
				const isProviderAvailable = Object.entries(payload.providers).some(
					([name, info]) => name === savedProvider && info.available
				);

				if (isProviderAvailable && savedProvider && savedModel) {
					console.log('Restoring valid saved provider/model:', savedProvider, savedModel);
					config.provider = savedProvider;
					config.modelName = savedModel;
				} else {
					// Find the first available provider
					const firstProvider = Object.entries(payload.providers).find(
						([_, info]) => info.available
					)?.[0];

					if (firstProvider) {
						console.log('Auto-selecting first available provider:', firstProvider);
						config.provider = firstProvider;
						// Auto-select first model if available
						const models = payload.providers[firstProvider].models;
						if (models && models.length > 0) {
							console.log('Auto-selecting first model:', models[0]);
							config.modelName = models[0];
						}
					}
				}
			} else if (type === 'settingsSaved') {
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
			}
		});
		return () => unsubscribe();
	});

	async function loadSettings() {
		messageBus.loadSettings();
	}

	async function saveSettings() {
		messageBus.saveSettings(config);
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
            showSettings = true;
            loadSettings();
        }}
        aria-label="Open Settings"
    >
        <span class="text-xl">&#9881;</span>
    </button>
</div>

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
                    <label for="openai-key" class="form-label">
                        OpenAI API Key
                    </label>
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
                    <label for="anthropic-key" class="form-label">
                        Anthropic API Key
                    </label>
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
                    <label for="ollama-host" class="form-label">
                        Ollama Host
                    </label>
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
                    <label for="provider-select" class="form-label">
                        Provider
                    </label>
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
                        <label for="model-name" class="form-label">
                            Model Name
                        </label>
                        <select
                            id="model-name"
                            bind:value={config.modelName}
                            class="form-input"
                        >
                            <option value="">Select Model</option>
                            {#each availableModels as model}
                                <option value={model}>{model}</option>
                            {/each}
                        </select>
                    </div>
                {/if}

                <div class="form-actions">
                    <button
                        type="button"
                        class="button-cancel"
                        onclick={() => (showSettings = false)}
                    >
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
