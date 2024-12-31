<script>
    let searchQuery = '';
    let searchResults = [];

    async function handleSearch() {
        // For this example, we'll just create dummy results
        // In a real app, you would make an API call here
        searchResults = [
            { title: `Result 1 for "${searchQuery}"`, url: '#' },
            { title: `Result 2 for "${searchQuery}"`, url: '#' },
            { title: `Result 3 for "${searchQuery}"`, url: '#' }
        ];
    }
</script>

<main class="min-h-screen bg-gray-50 px-4 py-12">
    <div class="mx-auto max-w-3xl">
        <h1 class="mb-8 text-center text-4xl font-bold text-gray-900">Search Engine</h1>
        
        <div class="flex w-full items-center gap-2 rounded-lg bg-white p-2 shadow-lg">
            <input
                type="text"
                bind:value={searchQuery}
                placeholder="Enter your search..."
                class="flex-1 rounded-md border-0 px-4 py-3 text-gray-900 outline-none ring-1 ring-gray-200 placeholder:text-gray-400 focus:ring-2 focus:ring-blue-500"
                on:keydown={(e) => e.key === 'Enter' && handleSearch()}
            />
            <button 
                on:click={handleSearch}
                class="rounded-md bg-blue-600 px-6 py-3 font-medium text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
                Search
            </button>
        </div>

        {#if searchResults.length > 0}
            <div class="mt-8 space-y-4">
                {#each searchResults as result}
                    <div class="rounded-lg bg-white p-4 shadow-sm transition-all hover:shadow-md">
                        <a href={result.url} class="block">
                            <h2 class="text-lg font-medium text-blue-600 hover:underline">{result.title}</h2>
                            <p class="mt-1 text-sm text-gray-600">{result.url}</p>
                        </a>
                    </div>
                {/each}
            </div>
        {/if}
    </div>
</main>
