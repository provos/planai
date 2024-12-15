import { jest, describe, afterEach, beforeEach, it, expect } from '@jest/globals';

// Create a mock Chart instance that we can reference
const mockChartInstance = {
    update: jest.fn(),
    data: {
        labels: [],
        datasets: [
            { data: [], label: 'Min', backgroundColor: 'rgba(75, 192, 192, 0.6)' },
            { data: [], label: 'Median', backgroundColor: 'rgba(54, 162, 235, 0.6)' },
            { data: [], label: 'Max', backgroundColor: 'rgba(255, 99, 132, 0.6)' },
            { data: [], label: 'Std Dev', backgroundColor: 'rgba(255, 206, 86, 0.6)' }
        ]
    }
};

// Mock global Chart constructor to return our instance
globalThis.Chart = jest.fn(() => mockChartInstance);

import { initWorkerStatsListeners, updateWorkerStats } from '../../../src/planai/static/js/workers.js';


describe('Worker Stats', () => {
    let workerStatsElement;
    let mockCanvas;

    beforeEach(() => {
        // Setup DOM elements
        workerStatsElement = document.createElement('div');
        workerStatsElement.id = 'worker-stats';
        document.body.appendChild(workerStatsElement);

        // Create real canvas element
        mockCanvas = document.createElement('canvas');
        mockCanvas.getContext = jest.fn().mockReturnValue({});

        // Spy on createElement to return our mockCanvas
        jest.spyOn(document, 'createElement').mockImplementation((tagName) => {
            if (tagName === 'canvas') {
                return mockCanvas;
            }
            return document.createElement(tagName);
        });

        // Clear mocks
        Chart.mockClear();
        mockChartInstance.update.mockClear();
    });

    afterEach(() => {
        document.body.innerHTML = '';
        jest.restoreAllMocks();
    });

    test('initializes chart with correct configuration', () => {
        const mockStats = {
            'Worker1': { min: 1, median: 2, max: 3, stdDev: 0.5 }
        };

        updateWorkerStats(mockStats);

        expect(Chart).toHaveBeenCalledTimes(1);
        expect(Chart.mock.calls[0][1]).toMatchObject({
            type: 'bar',
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    });

    test('updates chart with worker statistics', () => {
        const mockStats = {
            'Worker1': { min: 1, median: 2, max: 3, stdDev: 0.5 },
            'Worker2': { min: 2, median: 3, max: 4, stdDev: 0.7 }
        };

        updateWorkerStats(mockStats);

        expect(mockChartInstance.data.labels).toEqual(['Worker1', 'Worker2']);
        expect(mockChartInstance.data.datasets[0].data).toEqual([1, 2]); // min
        expect(mockChartInstance.data.datasets[1].data).toEqual([2, 3]); // median
        expect(mockChartInstance.data.datasets[2].data).toEqual([3, 4]); // max
        expect(mockChartInstance.data.datasets[3].data).toEqual([0.5, 0.7]); // stdDev
        expect(mockChartInstance.update).toHaveBeenCalled();
    });

    test('handles empty worker statistics', () => {
        const mockStats = {};
        updateWorkerStats(mockStats);

        expect(mockChartInstance.data.labels).toEqual([]);
        expect(mockChartInstance.data.datasets[0].data).toEqual([]);
        expect(mockChartInstance.update).toHaveBeenCalled();
    });

    test('event listener updates worker stats', () => {
        initWorkerStatsListeners();
        const mockStats = {
            'Worker1': { min: 1, median: 2, max: 3, stdDev: 0.5 }
        };

        window.dispatchEvent(new CustomEvent('statsUpdated', {
            detail: mockStats
        }));

        expect(mockChartInstance.data.labels).toEqual(['Worker1']);
        expect(mockChartInstance.update).toHaveBeenCalled();
    });
});