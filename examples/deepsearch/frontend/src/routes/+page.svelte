<script>
    import { onMount, onDestroy } from 'svelte';
    import { io } from 'socket.io-client';
    
    let searchQuery = '';
    let searchResults = [];
    let socket;
    let isLoading = false;
    let error = null;

    onMount(() => {
        socket = io('http://localhost:5050', {
            transports: ['websocket'],
            reconnection: true
        });
        
        socket.on('connect', () => {
            console.log('Connected to WebSocket server');
            error = null;
        });

        socket.on('connect_error', (err) => {
            console.error('Connection error:', err);
            error = 'Failed to connect to server';
        });
        
        socket.on('search_results', (results) => {
            console.log('Received search results:', results);
            isLoading = false;
            if (results.error) {
                error = results.error;
                return;
            }
            searchResults = results;
            error = null;
        });

        socket.on('search_error', (err) => {
            console.error('Search error:', err);
            isLoading = false;
            error = err;
        });
    });

    onDestroy(() => {
        if (socket) {
            socket.disconnect();
        }
    });

    async function handleSearch() {
        if (socket && searchQuery.trim()) {
            isLoading = true;
            error = null;
            searchResults = [];
            console.log('Emitting search:', searchQuery);
            socket.emit('search', searchQuery);
        }
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

        {#if error}
            <div class="mt-4 rounded-lg bg-red-100 p-4 text-red-700">
                {error}
            </div>
        {/if}

        {#if isLoading}
            <div class="mt-8 text-center text-gray-600">
                Searching...
            </div>
        {/if}

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
