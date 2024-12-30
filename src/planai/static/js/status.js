export function initStatusListeners() {
    createStatusInterface();
    window.addEventListener('statsUpdated', (e) => updateStatus(e.detail));
    window.addEventListener('logsUpdated', (e) => { updateLogs(e.detail) });
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

function createStatusInterface() {
    const container = document.getElementById('worker-status');

    const workerContainer = document.createElement('div');
    workerContainer.className = 'worker-container';

    // Create stats section
    const statsSection = document.createElement('div');
    statsSection.className = 'stats-section';
    statsSection.id = 'stats-section';

    // Create log console
    const logConsole = document.createElement('div');
    logConsole.className = 'log-console';
    logConsole.id = 'log-console';

    // Add both sections to container
    workerContainer.appendChild(statsSection);
    workerContainer.appendChild(logConsole);
    container.appendChild(workerContainer);
}

function updateStatus(workerStats) {
    const statsSection = document.getElementById('stats-section');
    statsSection.innerHTML = ''; // Clear previous state

    // Add worker rows to stats section
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

        statsSection.appendChild(row);
    });
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

const MAX_LOG_LINES = 100;

function updateLogs(logs) {
    const console = document.getElementById('log-console');
    if (!console) return;

    const wasScrolledToBottom = isScrolledToBottom(console);

    logs.forEach(log => {
        const entry = document.createElement('div');
        entry.className = 'log-entry';

        const timestamp = document.createElement('span');
        timestamp.className = 'log-timestamp';
        timestamp.textContent = new Date(log.timestamp * 1000).toLocaleTimeString();

        const message = document.createElement('span');
        message.className = 'log-message';
        const escapedMessage = escapeHtml(log.message);
        message.innerHTML = linkifyText(escapedMessage);

        entry.appendChild(timestamp);
        entry.appendChild(message);
        console.appendChild(entry);

        // Remove oldest entries if exceeding maximum
        while (console.children.length > MAX_LOG_LINES) {
            console.removeChild(console.firstChild);
        }
    });

    if (wasScrolledToBottom) {
        console.scrollTop = console.scrollHeight;
    }
}

function isScrolledToBottom(element) {
    const threshold = 1;
    return Math.abs(element.scrollHeight - element.clientHeight - element.scrollTop) < threshold;
}

function linkifyText(text) {
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    return text.replace(urlRegex, url => `<a href="${url}" target="_blank">${url}</a>`);
}