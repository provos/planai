let eventSource;
let reconnectAttempt = 0;

export function setupEventSource() {
    eventSource = new EventSource("/stream");
    eventSource.onmessage = handleServerMessage;
    eventSource.onerror = handleServerError;
}

function handleServerMessage(event) {
    const data = JSON.parse(event.data);
    // Dispatch to other modules
    window.dispatchEvent(new CustomEvent('tasksUpdated', { detail: data.tasks }));
    window.dispatchEvent(new CustomEvent('traceUpdated', { detail: data.trace }));
    window.dispatchEvent(new CustomEvent('statsUpdated', { detail: data.stats }));
    window.dispatchEvent(new CustomEvent('userRequestsUpdated', { detail: data.user_requests }));
    window.dispatchEvent(new CustomEvent('memoryUpdated', { detail: data.memory }));
    window.dispatchEvent(new CustomEvent('logsUpdated', { detail: data.logs }));
    window.dispatchEvent(new CustomEvent('graphsUpdated', { detail: data.graphs }));
    reconnectAttempt = 0;
}

function handleServerError(error) {
    eventSource.close();
    const timeout = Math.min(16000, Math.pow(2, reconnectAttempt) * 1000);
    reconnectAttempt++;
    setTimeout(setupEventSource, timeout);
}