let currentGraph = null;

export function initMermaid() {
    // Initialize mermaid
    mermaid.initialize({ startOnLoad: false });

    // Add graph update listener
    document.querySelector('[data-target="#worker-graph"]').addEventListener('click', updateGraph);

    // Export for use by other modules
    window.refreshGraph = updateGraph;
}

async function updateGraph() {
    const element = document.getElementById('mermaid-graph');
    if (!element) return;

    try {
        const response = await fetch('/graph');
        const data = await response.json();

        // Only update if the graph has changed
        if (element.querySelector('svg') && currentGraph === data.graph) {
            return;
        }

        currentGraph = data.graph;
        element.removeAttribute("data-processed");
        element.innerHTML = data.graph;
        await mermaid.init(undefined, element);
    } catch (error) {
        console.error('Error updating graph:', error);
    }
}