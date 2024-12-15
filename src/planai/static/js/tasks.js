import { formatTime, renderObject } from './utils.js';

export function initTaskListeners() {
    window.addEventListener('tasksUpdated', (e) => updateTaskLists(e.detail));
    window.addEventListener('userRequestsUpdated', (e) => updateUserRequests(e.detail));
}

function updateTaskLists(data) {
    updateTaskList('queued-tasks', data.queued);
    updateTaskList('active-tasks', data.active);
    updateTaskList('completed-tasks', data.completed);
    updateTaskList('failed-tasks', data.failed);
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
    <div class="task-item" style="${borderStyle}">
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

        // We need to directly reference the toogleProvenance function
        const taskItemDiv = taskElement.querySelector('.task-item');
        taskItemDiv.addEventListener('click', () => toggleProvenance(task.id));
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


function toggleProvenance(taskId) {
    console.log('Toggling provenance for task', taskId);
    const provenanceElement = document.getElementById(`provenance-${taskId}`);
    if (provenanceElement) {
        const newDisplay = provenanceElement.style.display === 'block' ? 'none' : 'block';
        provenanceElement.style.display = newDisplay;
        provenanceVisibility[taskId] = newDisplay === 'block';
    }
}


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
