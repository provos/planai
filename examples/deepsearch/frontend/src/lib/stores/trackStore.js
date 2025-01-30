import { writable } from 'svelte/store';

export const tracks = [
    {
        src: 'https://sveltejs.github.io/assets/music/strauss.mp3',
        title: 'The Blue Danube Waltz',
        artist: 'Johann Strauss'
    },
    {
        src: 'https://sveltejs.github.io/assets/music/holst.mp3',
        title: 'Mars, the Bringer of War',
        artist: 'Gustav Holst'
    },
    {
        src: 'https://sveltejs.github.io/assets/music/satie.mp3',
        title: 'Gymnopédie no. 1',
        artist: 'Erik Satie'
    },
    {
        src: 'https://sveltejs.github.io/assets/music/mozart.mp3',
        title: 'Requiem in D minor, K. 626 - III. Sequence - Lacrymosa',
        artist: 'Wolfgang Amadeus Mozart'
    }
];

export const currentTrack = writable(tracks[0]);
