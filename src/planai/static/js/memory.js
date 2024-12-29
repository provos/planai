export function initMemoryListeners() {
    window.addEventListener('memoryUpdated', (e) => updateMemoryStats(e.detail));
}

function updateMemoryStats(memoryData) {
    document.getElementById('current-memory').textContent = memoryData.current;
    document.getElementById('avg-memory').textContent = memoryData.average;
    document.getElementById('peak-memory').textContent = memoryData.peak;
}

