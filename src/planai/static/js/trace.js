export function initTraceListeners() {
    window.addEventListener('traceUpdated', (e) => updateTraceVisualization(e.detail));
}

function updateTraceVisualization(traceData) {
    const traceElement = document.getElementById('trace-visualization');
    const expandedPrefixes = new Set();

    // Store currently expanded prefixes
    traceElement.querySelectorAll('.trace-prefix.expanded').forEach(el => {
        expandedPrefixes.add(el.querySelector('h4').textContent.trim());
    });

    traceElement.innerHTML = '';

    for (const [prefixStr, entries] of Object.entries(traceData)) {
        const prefixElement = document.createElement('div');
        prefixElement.className = 'trace-prefix';

        const prefix = prefixStr.split('_').join(', ');

        const headerElement = document.createElement('h4');
        headerElement.textContent = `Prefix: (${prefix})`;
        headerElement.onclick = function () {
            this.parentElement.classList.toggle('expanded');
        };

        prefixElement.appendChild(headerElement);

        const tableWrapper = document.createElement('div');
        tableWrapper.className = 'table-wrapper';

        const table = document.createElement('table');
        table.className = 'trace-table';

        // Create table header
        const headerRow = table.insertRow();
        ['Worker', 'Action', 'Task', 'Count', 'Status'].forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });

        // Populate table with entry data
        entries.forEach(entry => {
            const row = table.insertRow();
            row.className = entry.action;
            if (entry.count === 0) row.classList.add('zero-count');
            if (entry.status.includes('Notifying:')) row.classList.add('notification-row');

            ['worker', 'action', 'task', 'count'].forEach(key => {
                const cell = row.insertCell();
                cell.textContent = entry[key];
            });

            // Handle status separately to allow for wrapping
            const statusCell = row.insertCell();
            statusCell.className = 'status-cell';
            statusCell.textContent = entry.status;
        });

        tableWrapper.appendChild(table);
        prefixElement.appendChild(tableWrapper);
        traceElement.appendChild(prefixElement);

        // Restore expanded state if it was previously expanded
        if (expandedPrefixes.has(headerElement.textContent.trim())) {
            prefixElement.classList.add('expanded');
        }
    }
}