export function formatTime(seconds) {
    if (seconds === null || isNaN(seconds)) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${remainingSeconds}s`;
}

export function renderObject(obj, indent = '') {
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

