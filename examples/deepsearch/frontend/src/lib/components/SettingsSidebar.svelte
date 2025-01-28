<script>
	import { FontAwesomeIcon } from '@fortawesome/svelte-fontawesome';
	import { faPaperPlane, faStop } from '@fortawesome/free-solid-svg-icons';
    import { messageBus } from '../stores/messageBus.svelte.js';

    const providerMapping = {
        'ollama': 'Ollama',
        'openai': 'OpenAI',
        'anthropic': 'Anthropic'
    };

	let showSettings = $state(false);

	let config = $state({
		serperApiKey: '',
		openAiApiKey: '',
		anthropicApiKey: '',
		provider: 'Ollama',
		modelName: ''
	});

	let hasKey = $state({
		serper: false,
		openAi: false,
		anthropic: false
	});

    // Add effect to handle settings responses
    $effect(() => {
        const unsubscribe = messageBus.subscribe(({ type, payload }) => {
            if (type === 'settingsLoaded') {
                config.serperApiKey = payload.serperApiKey || '';
                config.openAiApiKey = payload.openAiApiKey || '';
                config.anthropicApiKey = payload.anthropicApiKey || '';
                // Map provider name from backend format to display format
                config.provider = providerMapping[payload.provider?.toLowerCase()] || 'Ollama';
                config.modelName = payload.modelName || '';
                hasKey.serper = !!payload.serperApiKey;
                hasKey.openAi = !!payload.openAiApiKey;
                hasKey.anthropic = !!payload.anthropicApiKey;
            } else if (type === 'settingsSaved') {
                showSettings = false;
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
</script>

<div class="fixed top-0 left-0 h-screen w-16 flex flex-col items-center bg-gray-200 dark:bg-gray-800 shadow-lg">
	<!-- Gear Icon -->
	<button
		class="mt-4 text-gray-700 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400 transition-colors relative"
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
        class="fixed inset-0 bg-black bg-opacity-50 z-50 w-full h-full flex"
        aria-labelledby="settings-title"
        open
    >
        <button
            class="absolute inset-0 w-full h-full cursor-default"
            onclick={() => (showSettings = false)}
            onkeydown={(e) => {
                if (e.key === 'Escape') showSettings = false;
            }}
        >
            <span class="sr-only">Close settings</span>
        </button>

        <section
            class="absolute left-20 top-4 bg-white dark:bg-gray-800 rounded-lg p-6 w-96 shadow-lg"
            role="document"
        >
            <h2 id="settings-title" class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Settings</h2>
            <form
                onsubmit={(e) => {
                    e.preventDefault();
                    saveSettings();
                }}
                class="space-y-4"
            >
                <div>
					<label for="serper-key" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
						>Serper API Key <span class="text-red-500">*</span></label
					>
					<input
						id="serper-key"
						type="text"
						bind:value={config.serperApiKey}
						placeholder={hasKey.serper ? 'Key stored' : 'Enter your Serper key'}
						class="w-full rounded-md border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
						required
					/>
				</div>
				<div>
						<label for="openai-key" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
							>OpenAI API Key</label
						>
						<input
							id="openai-key"
							type="text"
							bind:value={config.openAiApiKey}
							placeholder={hasKey.openAi ? 'Key stored' : 'Enter your OpenAI key'}
							class="w-full rounded-md border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
						/>
					</div>
					<div>
						<label for="anthropic-key" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
							>Anthropic API Key</label
						>
						<input
							id="anthropic-key"
							type="text"
							bind:value={config.anthropicApiKey}
							placeholder={hasKey.anthropic ? 'Key stored' : 'Enter your Anthropic key'}
							class="w-full rounded-md border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
						/>
					</div>
					<div>
						<label for="provider-select" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
							>Provider</label
						>
						<select
							id="provider-select"
							bind:value={config.provider}
							class="w-full rounded-md border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
						>
							<option value="Ollama">Ollama</option>
							<option value="OpenAI">OpenAI</option>
							<option value="Anthropic">Anthropic</option>
						</select>
					</div>
					<div>
						<label for="model-name" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
							>Model Name</label
						>
						<input
							id="model-name"
							type="text"
							bind:value={config.modelName}
							placeholder="e.g. llama2"
							class="w-full rounded-md border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
						/>
					</div>
				<div class="flex justify-end space-x-2 mt-4">
					<button
						type="button"
						class="px-4 py-2 rounded-md bg-gray-200 hover:bg-gray-300 text-gray-800 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-200 transition-colors"
						onclick={() => (showSettings = false)}
					>
						Cancel
					</button>
					<button
						type="submit"
						class="px-4 py-2 rounded-md bg-blue-600 hover:bg-blue-700 text-white dark:bg-blue-500 dark:hover:bg-blue-600 transition-colors"
					>
						Save
					</button>
				</div>
            </form>
        </section>
    </dialog>
{/if}