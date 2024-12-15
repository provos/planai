import { setupEventSource } from './core.js';
import { initTaskListeners } from './tasks.js';
import { initTraceListeners } from './trace.js';
import { initWorkerStatsListeners } from './workers.js';
import { initTheme } from './theme.js';

setupEventSource();
initTaskListeners();
initTraceListeners();
initWorkerStatsListeners();
initTheme();

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