export function initStatusListeners() {
    window.addEventListener('statsUpdated', (e) => updateStatus(e.detail));
}

function createStatusBar(count, type) {
    const bar = document.createElement('div');
    bar.className = `status-${type}`;
    bar.style.display = 'flex';
    for (let i = 0; i < count; i++) {
        const square = document.createElement('div');
        square.className = `status-square ${type}`;
        bar.appendChild(square);
    }
    return bar;
}

function updateStatus(workerStats) {
    const container = document.getElementById('worker-status');
    container.innerHTML = ''; // Clear previous state

    const maxWidth = 50; // Maximum number of squares to show

    Object.entries(workerStats).forEach(([worker, stats]) => {
        const row = document.createElement('div');
        row.className = 'worker-row';

        // Worker name
        const name = document.createElement('div');
        name.className = 'worker-name';
        name.textContent = worker;
        row.appendChild(name);

        // Status bars container
        const bars = document.createElement('div');
        bars.className = 'status-bars';

        // Calculate scaling
        const total = stats.completed + stats.active + stats.queued + stats.failed;
        const scale = total > maxWidth ? maxWidth / total : 1;

        // Add status bars
        if (stats.completed > 0) bars.appendChild(createStatusBar(Math.max(1, Math.floor(stats.completed * scale)), 'completed'));
        if (stats.active > 0) bars.appendChild(createStatusBar(Math.max(1, Math.floor(stats.active * scale)), 'active'));
        if (stats.queued > 0) bars.appendChild(createStatusBar(Math.max(1, Math.floor(stats.queued * scale)), 'queued'));
        if (stats.failed > 0) bars.appendChild(createStatusBar(Math.max(1, Math.floor(stats.failed * scale)), 'failed'));

        row.appendChild(bars);

        // Statistics
        const stats_div = document.createElement('div');
        stats_div.className = 'worker-stats';
        stats_div.textContent = `C:${stats.completed} A:${stats.active} Q:${stats.queued} F:${stats.failed}`;
        row.appendChild(stats_div);

        container.appendChild(row);
    });
}