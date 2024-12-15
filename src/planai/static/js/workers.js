// Execution statistics chart

export function initWorkerStatsListeners() {
    window.addEventListener('statsUpdated', (e) => updateWorkerStats(e.detail));
}

let workerChart;

export function updateWorkerStats(workerStats) {
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
