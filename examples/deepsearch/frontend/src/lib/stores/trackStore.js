import { writable } from 'svelte/store';

export const tracks = [
    {
        src: 'https://sveltejs.github.io/assets/music/strauss.mp3',
        title: 'The Blue Danube Waltz',
        artist: 'Johann Strauss',
        info: 'One of the most famous waltzes ever written, composed in 1866.',
        link: 'https://en.wikipedia.org/wiki/The_Blue_Danube'
    },
    {
        src: 'https://sveltejs.github.io/assets/music/holst.mp3',
        title: 'Mars, the Bringer of War',
        artist: 'Gustav Holst',
        info: 'First movement of The Planets suite, composed between 1914 and 1916.',
        link: 'https://en.wikipedia.org/wiki/The_Planets'
    },
    {
        src: 'https://sveltejs.github.io/assets/music/satie.mp3',
        title: 'Gymnopédie no. 1',
        artist: 'Erik Satie',
        info: 'A gentle, atmospheric piece that defined ambient music, published in 1888.',
        link: 'https://en.wikipedia.org/wiki/Gymnop%C3%A9dies'
    },
    {
        src: 'https://sveltejs.github.io/assets/music/mozart.mp3',
        title: 'Requiem in D minor, K. 626 - III. Sequence - Lacrymosa',
        artist: 'Wolfgang Amadeus Mozart',
        info: 'Part of Mozart\'s final composition, left unfinished at his death in 1791.',
        link: 'https://en.wikipedia.org/wiki/Requiem_(Mozart)'
    }
];

export const currentTrack = writable(tracks[0]);
