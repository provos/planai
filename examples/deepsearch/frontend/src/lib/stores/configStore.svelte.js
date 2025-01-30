import { writable } from 'svelte/store';

export const configState = writable({
    isValid: false,
    provider: '',
    modelName: ''
});