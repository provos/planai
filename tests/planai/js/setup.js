import { jest } from '@jest/globals';

// Setup test environment with ES modules
import 'jest-environment-jsdom';

// Add global mocks
globalThis.Chart = jest.fn();

// Mock window.EventSource
globalThis.EventSource = jest.fn(() => ({
    onmessage: jest.fn(),
    onerror: jest.fn(),
    close: jest.fn()
}));

// Add custom matchers
expect.extend({
    // Add custom matchers if needed
});