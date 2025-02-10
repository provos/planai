export function initGraphListeners() {
    const graphSelect = document.getElementById('graph-select');
    if (!graphSelect) return;

    // Listen for graph updates from the event stream
    window.addEventListener('graphsUpdated', (e) => updateGraphs(e.detail));

    // Handle graph selection changes
    graphSelect.addEventListener('change', async (event) => {
        const graphId = parseInt(event.target.value, 10);
        try {
            const response = await fetch('/select_graph', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ graph_id: graphId }),
            });

            if (!response.ok) {
                throw new Error('Failed to select graph');
            }

            // Trigger a refresh of the mermaid graph
            window.refreshGraph && window.refreshGraph();

        } catch (error) {
            console.error('Error selecting graph:', error);
        }
    });
}

function updateGraphs(graphs) {
    const graphSelect = document.getElementById('graph-select');
    if (!graphSelect || !graphs || !Array.isArray(graphs)) return;

    // Store current selection
    const currentValue = graphSelect.value;

    // Clear existing options
    graphSelect.innerHTML = '';

    // Add new options
    graphs.forEach(graph => {
        const option = document.createElement('option');
        option.value = graph.id;
        option.textContent = graph.name;
        graphSelect.appendChild(option);
    });

    // Restore selection if it still exists
    if (graphs.some(g => g.id.toString() === currentValue)) {
        graphSelect.value = currentValue;
    }
}
