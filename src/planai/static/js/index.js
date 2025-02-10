import { setupEventSource } from './core.js';
import { initTaskListeners } from './tasks.js';
import { initTraceListeners } from './trace.js';
import { initWorkerStatsListeners } from './workers.js';
import { initStatusListeners } from './status.js';
import { initTheme } from './theme.js';
import { initTabs } from './utils.js';
import { initMermaid } from './mermaid.js';
import { initMemoryListeners } from './memory.js';
import { initGraphListeners } from './graphs.js';

setupEventSource();
initTaskListeners();
initTraceListeners();
initWorkerStatsListeners();
initStatusListeners();
initTheme();
initTabs();
initMermaid();
initMemoryListeners();
initGraphListeners();

// Quit button functionality
document.getElementById('quit-button').addEventListener('click', function () {
    fetch('/quit', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'ok') {
                alert('Quit signal sent. The application will shut down soon.');
            }
        });
});