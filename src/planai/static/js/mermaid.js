export function initMermaid() {
    // Initialize mermaid
    mermaid.initialize({ startOnLoad: false });

    // Add graph update listener
    document.querySelector('[data-target="#worker-graph"]').addEventListener('click', updateGraph);
}

async function updateGraph() {
    const element = document.getElementById('mermaid-graph');

    // Check if the element has already been rendered by mermaid
    if (element && element.querySelector('svg')) {
        // If it has, we don't need to re-render it - can be removed if the graph changes during execution
        return;
    }

    const response = await fetch('/graph');
    const data = await response.json();

    element.removeAttribute("data-processed");
    element.innerHTML = data.graph;

    mermaid.init(undefined, element);
}