export function initGraphListeners() {
    const graphSelect = document.getElementById('graph-select');
    if (!graphSelect) return;

    // Listen for graph updates from the event stream
    window.addEventListener('graphsUpdated', (e) => updateGraphs(e.detail));

    // Handle graph selection changes
    graphSelect.addEventListener('change', async (event) => {
        const value = event.target.value;
        if (!value) return;  // Don't proceed if no valid value

        const graphId = parseInt(value, 10);
        if (isNaN(graphId)) return;  // Don't proceed if not a number

        try {
            const response = await fetch('/select_graph', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ graph_id: graphId })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Failed to select graph');
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

    // Store current selection if it exists and is valid
    const currentValue = graphSelect.value;
    const hasValidSelection = currentValue &&
        graphs.some(g => g.index !== undefined &&
            g.index.toString() === currentValue);

    // Clear existing options
    graphSelect.innerHTML = '';

    // Add new options
    graphs.forEach(graph => {
        if (graph && graph.index !== undefined && graph.name !== undefined) {
            const option = document.createElement('option');
            option.value = graph.index.toString();  // Ensure it's a string
            option.textContent = graph.name;
            graphSelect.appendChild(option);
        }
    });

    // Restore selection if it was valid
    if (hasValidSelection) {
        graphSelect.value = currentValue;
    } else if (graphs.length > 0 && graphs[0].id !== undefined) {
        // Select first graph if available
        graphSelect.value = graphs[0].id.toString();
    }
}
