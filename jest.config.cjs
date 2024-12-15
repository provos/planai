module.exports = {
    testEnvironment: 'jsdom',
    roots: ['<rootDir>/tests/planai/js'],
    moduleDirectories: ['node_modules', 'static/js'],
    setupFilesAfterEnv: ['<rootDir>/tests/planai/js/setup.js'],
    transform: {},
    testMatch: [
        "**/__tests__/**/*.[jt]s?(x)",
        "**/?(*.)+(spec|test).[tj]s?(x)",
        "**/?(*.)+(spec|test).cjs"
    ]
};