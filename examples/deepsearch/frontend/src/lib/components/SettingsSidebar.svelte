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
</script>

<div
	class="fixed left-0 top-0 flex h-screen w-16 flex-col items-center bg-gray-200 shadow-lg dark:bg-gray-800"
>
	<!-- Gear Icon -->
	<button
		class="relative mt-4 text-gray-700 transition-colors hover:text-blue-600 dark:text-gray-200 dark:hover:text-blue-400"
		onclick={() => {
			showSettings = true;
			loadSettings();
		}}
		aria-label="Open Settings"
	>
		<!-- <FontAwesomeIcon icon={faCog} class="w-6 h-6" /> -->
		<span class="text-xl">&#9881;</span>
	</button>
	<!-- ...any additional nav items... -->
</div>

{#if showSettings}
	<dialog
		class="fixed inset-0 z-50 flex h-full w-full bg-black bg-opacity-50"
		aria-labelledby="settings-title"
		open
	>
		<button
			class="absolute inset-0 h-full w-full cursor-default"
			onclick={() => (showSettings = false)}
			onkeydown={(e) => {
				if (e.key === 'Escape') showSettings = false;
			}}
		>
			<span class="sr-only">Close settings</span>
		</button>

		<section
			class="absolute left-20 top-4 w-96 rounded-lg bg-white p-6 shadow-lg dark:bg-gray-800"
			role="document"
		>
			<h2 id="settings-title" class="mb-4 text-xl font-semibold text-gray-900 dark:text-white">
				Settings
			</h2>
			<form
				onsubmit={(e) => {
					e.preventDefault();
					saveSettings();
				}}
				class="space-y-4"
			>
				<div>
					<label
						for="serper-key"
						class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
						>Serper API Key {#if !hasSerperKey}<span class="text-red-500">*</span>{/if}</label
					>
					<input
						id="serper-key"
						type="password"
						bind:value={config.serperApiKey}
						placeholder={hasSerperKey ? 'Key stored' : 'Enter your Serper key'}
						class="w-full rounded-md border-gray-300 bg-white px-3 py-2 text-gray-900 focus:border-transparent focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
						required={!hasSerperKey}
					/>
				</div>
				<div>
					<label
						for="openai-key"
						class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
						>OpenAI API Key</label
					>
					<input
						id="openai-key"
						type="password"
						bind:value={config.openAiApiKey}
						oninput={(e) => onApiKeyChange('openai', e.target.value)}
						placeholder={providers.openai?.hasKey ? 'Key stored' : 'Enter your OpenAI key'}
						class="w-full rounded-md border-gray-300 bg-white px-3 py-2 text-gray-900 focus:border-transparent focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
					/>
				</div>
				<div>
					<label
						for="anthropic-key"
						class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
						>Anthropic API Key</label
					>
					<input
						id="anthropic-key"
						type="password"
						bind:value={config.anthropicApiKey}
						oninput={(e) => onApiKeyChange('anthropic', e.target.value)}
						placeholder={providers.anthropic?.hasKey ? 'Key stored' : 'Enter your Anthropic key'}
						class="w-full rounded-md border-gray-300 bg-white px-3 py-2 text-gray-900 focus:border-transparent focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
					/>
				</div>
				<div>
					<label
						for="ollama-host"
						class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
						>Ollama Host</label
					>
					<input
						id="ollama-host"
						type="text"
						bind:value={config.ollamaHost}
						oninput={onOllamaHostChange}
						placeholder="localhost:11434"
						class="w-full rounded-md border-gray-300 bg-white px-3 py-2 text-gray-900 focus:border-transparent focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
					/>
				</div>
				<div>
					<label
						for="provider-select"
						class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300">Provider</label
					>
					<select
						id="provider-select"
						bind:value={config.provider}
						onchange={onProviderChange}
						class="w-full rounded-md border-gray-300 bg-white px-3 py-2 text-gray-900 focus:border-transparent focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
					>
						<option value="">Select Provider</option>
						{#each availableProviders as provider}
							<option value={provider}
								>{provider.charAt(0).toUpperCase() + provider.slice(1)}</option
							>
						{/each}
					</select>
				</div>
				{#if config.provider && providers[config.provider]?.models.length > 0}
					<div>
						<label
							for="model-name"
							class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
							>Model Name</label
						>
						<select
							id="model-name"
							bind:value={config.modelName}
							class="w-full rounded-md border-gray-300 bg-white px-3 py-2 text-gray-900 focus:border-transparent focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
						>
							<option value="">Select Model</option>
							{#each availableModels as model}
								<option value={model}>{model}</option>
							{/each}
						</select>
					</div>
				{/if}
				<div class="mt-4 flex justify-end space-x-2">
					<button
						type="button"
						class="rounded-md bg-gray-200 px-4 py-2 text-gray-800 transition-colors hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
						onclick={() => (showSettings = false)}
					>
						Cancel
					</button>
					<button
						type="submit"
						class="rounded-md px-4 py-2 transition-colors"
						class:bg-blue-600={canSave}
						class:hover:bg-blue-700={canSave}
						class:bg-gray-400={!canSave}
						class:cursor-not-allowed={!canSave}
						disabled={!canSave}
					>
						Save
					</button>
				</div>
			</form>
		</section>
	</dialog>
{/if}
