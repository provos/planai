export function formatTime(seconds) {
    if (seconds === null || isNaN(seconds)) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${remainingSeconds}s`;
}

// Helper function to escape HTML special characters
export function escapeHtml(unsafe) {
    if (unsafe === null || unsafe === undefined) return '';
    return String(unsafe)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

export function renderObject(obj, indent = '') {
    return Object.entries(obj).map(([key, value]) => {
        const escapedKey = escapeHtml(key);
        if (value === null) return `${indent}<strong>${escapedKey}:</strong> null`;
        if (typeof value === 'object') {
            return `
                ${indent}<strong>${escapedKey}:</strong>
                <div class="nested-object">
                    ${renderObject(value, indent + '  ')}
                </div>
            `;
        }
        return `${indent}<strong>${escapedKey}:</strong> ${escapeHtml(JSON.stringify(value))}`;
    }).join('<br>');
}

export function initTabs() {
    const tabs = document.querySelectorAll('.tab-button');
    const contents = document.querySelectorAll('.tab-content');

    // Show first tab by default
    tabs[0]?.classList.add('active');
    contents[0]?.classList.add('active');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked tab and corresponding content
            tab.classList.add('active');
            const content = document.querySelector(tab.dataset.target);
            content?.classList.add('active');
        });
    });
}