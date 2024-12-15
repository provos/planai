let eventSource;
let reconnectAttempt = 0;

function setupEventSource() {
    eventSource = new EventSource("/stream");

    eventSource.onmessage = function (event) {
        const data = JSON.parse(event.data);
        updateTaskLists(data.tasks);
        updateTraceVisualization(data.trace);
        updateWorkerStats(data.stats);
        updateUserRequests(data.user_requests);
        reconnectAttempt = 0;
    };

    eventSource.onerror = function (error) {
        eventSource.close();
        const timeout = Math.min(16000, Math.pow(2, reconnectAttempt) * 1000);
        reconnectAttempt++;
        setTimeout(setupEventSource, timeout);
    };
}

setupEventSource();

function updateTaskLists(data) {
    updateTaskList('queued-tasks', data.queued);
    updateTaskList('active-tasks', data.active);
    updateTaskList('completed-tasks', data.completed);
    updateTaskList('failed-tasks', data.failed);
}

function formatTime(seconds) {
    if (seconds === null || isNaN(seconds)) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${remainingSeconds}s`;
}

let provenanceVisibility = {};

function updateTaskList(elementId, tasks) {
    const element = document.getElementById(elementId);
    const now = Date.now() / 1000; // Current time in seconds

    tasks.forEach(task => {
        let taskElement = document.getElementById(`task-${task.id}`);
        if (!taskElement) {
            taskElement = document.createElement('div');
            taskElement.id = `task-${task.id}`;
            taskElement.className = 'task-item';
            element.appendChild(taskElement);
        }

        // Add a red border for failed tasks
        const isFailed = elementId === 'failed-tasks';
        const borderStyle = isFailed ? 'border: 2px solid red;' : '';

        let timeInfo = '';
        if (elementId === 'active-tasks') {
            const elapsedTime = task.start_time ? now - task.start_time : 0;
            timeInfo = `<br><strong>Elapsed Time:</strong> <span class="elapsed-time" data-start="${task.start_time}">${formatTime(elapsedTime)}</span>`;
        } else if (elementId === 'completed-tasks' && task.start_time && task.end_time) {
            const totalTime = task.end_time - task.start_time;
            timeInfo = `<br><strong>Total Time:</strong> ${formatTime(totalTime)}`;
        }

        const provenanceDisplay = provenanceVisibility[task.id] ? 'block' : 'none';

        taskElement.innerHTML = `
    <div class="task-item" onclick="toggleProvenance('${task.id}')" style="${borderStyle}">
        <strong>ID:</strong> ${task.id}<br>
        <strong>Type:</strong> ${task.type}<br>
        <strong>Worker:</strong> ${task.worker}
        ${timeInfo}
        ${isFailed ? `<br><strong>Error:</strong> ${task.error || 'Unknown error'}` : ''}
        <div id="provenance-${task.id}" class="provenance-info" style="display: ${provenanceDisplay};">
            <h4>Provenance:</h4>
            <ul>
                ${task.provenance.map(p => `<li>${p}</li>`).join('')}
            </ul>
            <h4>Input Provenance:</h4>
            <ul>
                ${task.input_provenance.map(inputTask => `
                    <li class="input-provenance-item">
                        ${renderObject(inputTask)}
                    </li>
                `).join('')}
            </ul>
        </div>
    </div>
`;
    });

    // Remove tasks that no longer exist
    Array.from(element.children).forEach(child => {
        const taskId = child.id.replace('task-', '');
        if (!tasks.some(task => task.id === taskId)) {
            element.removeChild(child);
        }
    });
}

function updateElapsedTime() {
    const now = Date.now() / 1000;
    document.querySelectorAll('.elapsed-time').forEach(element => {
        const startTime = parseFloat(element.getAttribute('data-start'));
        if (startTime) {
            const elapsedTime = now - startTime;
            element.textContent = formatTime(elapsedTime);
        }
    });
}

// Set up interval to update elapsed time every second
setInterval(updateElapsedTime, 1000);


function renderObject(obj, indent = '') {
    return Object.entries(obj).map(([key, value]) => {
        if (value === null) return `${indent}<strong>${key}:</strong> null`;
        if (typeof value === 'object') {
            return `
                ${indent}<strong>${key}:</strong>
                <div class="nested-object">
                    ${renderObject(value, indent + '  ')}
                </div>
            `;
        }
        return `${indent}<strong>${key}:</strong> ${JSON.stringify(value)}`;
    }).join('<br>');
}

function toggleProvenance(taskId) {
    console.log('Toggling provenance for task', taskId);
    const provenanceElement = document.getElementById(`provenance-${taskId}`);
    if (provenanceElement) {
        const newDisplay = provenanceElement.style.display === 'block' ? 'none' : 'block';
        provenanceElement.style.display = newDisplay;
        provenanceVisibility[taskId] = newDisplay === 'block';
    }
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

// Execution statistics chart
let workerChart;

function updateWorkerStats(workerStats) {
    const workerStatsElement = document.getElementById('worker-stats');

    if (!workerChart) {
        // Create the chart if it doesn't exist
        const canvas = document.createElement('canvas');
        canvas.id = 'workerStatsChart';
        workerStatsElement.appendChild(canvas);
        const ctx = canvas.getContext('2d');

        workerChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Min',
                        data: [],
                        backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    },
                    {
                        label: 'Median',
                        data: [],
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    },
                    {
                        label: 'Max',
                        data: [],
                        backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    },
                    {
                        label: 'Std Dev',
                        data: [],
                        backgroundColor: 'rgba(255, 206, 86, 0.6)',
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Execution Time (seconds)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Worker Execution Statistics'
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y.toFixed(2) + ' seconds';
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }

    // Update chart data
    const labels = Object.keys(workerStats);
    const minData = labels.map(worker => workerStats[worker].min);
    const medianData = labels.map(worker => workerStats[worker].median);
    const maxData = labels.map(worker => workerStats[worker].max);
    const stdDevData = labels.map(worker => workerStats[worker].stdDev);

    workerChart.data.labels = labels;
    workerChart.data.datasets[0].data = minData;
    workerChart.data.datasets[1].data = medianData;
    workerChart.data.datasets[2].data = maxData;
    workerChart.data.datasets[3].data = stdDevData;

    workerChart.update();
}


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

// Theme toggle functionality
const themeToggle = document.getElementById('theme-toggle');
const prefersDarkScheme = window.matchMedia("(prefers-color-scheme: dark)");

function setTheme(theme) {
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
}

// Check for saved theme preference or use the system preference
const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
    setTheme(savedTheme);
} else if (prefersDarkScheme.matches) {
    setTheme('dark');
}

// Toggle theme when button is clicked
themeToggle.addEventListener('click', () => {
    const currentTheme = document.body.getAttribute('data-theme');
    setTheme(currentTheme === 'dark' ? 'light' : 'dark');
});

// User input requests
function updateUserRequests(userRequests) {
    const requestsContainer = document.getElementById('user-input-requests');
    const existingTiles = requestsContainer.getElementsByClassName('user-request-tile');

    // Map existing tiles by data-task-id
    const existingTilesMap = {};
    Array.from(existingTiles).forEach(tile => {
        const taskId = tile.getAttribute('data-task-id');
        existingTilesMap[taskId] = tile;
    });

    // Set of current user request IDs
    const userRequestIds = new Set(userRequests.map(request => request.task_id.toString()));

    // Add new request tiles
    userRequests.forEach(request => {
        const taskId = request.task_id.toString();
        if (!existingTilesMap[taskId]) {
            // Create new tile
            const requestTile = document.createElement('div');
            requestTile.className = 'user-request-tile';
            requestTile.setAttribute('data-task-id', taskId);
            requestTile.setAttribute('data-accepted-mime-types', request.accepted_mime_types.join(','));

            requestTile.innerHTML = `
        <p><strong>Instruction:</strong> ${request.instruction}</p>
        <p>Drag a file here to upload or</p>
        <button class="abort-button">Abort</button>
    `;

            requestTile.addEventListener('dragover', function (event) {
                event.preventDefault();
                requestTile.classList.add('drag-over');
            });

            requestTile.addEventListener('dragleave', function () {
                requestTile.classList.remove('drag-over');
            });

            requestTile.addEventListener('drop', handleFileDrop);

            const abortButton = requestTile.querySelector('.abort-button');
            abortButton.addEventListener('click', () => abortRequest(request.task_id));

            requestsContainer.appendChild(requestTile);
        }
        // Existing requests are immutable; no updates needed
    });

    // Remove tiles that are no longer in userRequests
    Array.from(existingTiles).forEach(tile => {
        const taskId = tile.getAttribute('data-task-id');
        if (!userRequestIds.has(taskId)) {
            requestsContainer.removeChild(tile);
        }
    });
}

function handleFileDrop(event) {
    event.preventDefault();
    const requestTile = event.currentTarget;
    requestTile.classList.remove('drag-over');

    const taskId = requestTile.getAttribute('data-task-id');
    const acceptedMimeTypes = requestTile.getAttribute('data-accepted-mime-types').split(',');

    const files = event.dataTransfer.files;
    if (files.length === 0) return;

    const file = files[0];
    if (!acceptedMimeTypes.includes(file.type)) {
        alert('Invalid file type. Please provide a file of type: ' + acceptedMimeTypes.join(', '));
        return;
    }

    const formData = new FormData();
    formData.append('task_id', taskId);
    formData.append('file', file);

    fetch('/user_input', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'ok') {
                requestTile.remove(); // Remove the tile after successful upload
            }
        })
        .catch(error => {
            console.error('Error uploading file:', error);
        });
}

function abortRequest(taskId) {
    if (confirm('Are you sure you want to abort this request?')) {
        const formData = new FormData();
        formData.append('task_id', taskId);
        formData.append('abort', 'true');  // Use a string 'true' since FormData values are strings

        fetch('/user_input', {
            method: 'POST',
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'ok') {
                    const requestTile = document.querySelector(`.user-request-tile[data-task-id="${taskId}"]`);
                    if (requestTile) {
                        requestTile.remove();
                    }
                } else {
                    alert('Failed to abort the request.');
                }
            })
            .catch(error => {
                console.error('Error aborting request:', error);
            });
    }
}
