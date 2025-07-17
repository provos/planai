// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
    site: 'https://docs.getplanai.com/',
    integrations: [starlight({
        title: 'PlanAI',
        description: 'Graph-based AI workflow automation framework',
        social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/provos/planai' }],
        sidebar: [
            {
                label: 'Getting Started',
                items: [
                    { label: 'Installation', slug: 'getting-started/installation' },
                    { label: 'Quick Start', slug: 'getting-started/quickstart' },
                ],
            },
            {
                label: 'User Guide',
                items: [
                    { label: 'Basic Usage', slug: 'guide/usage' },
                    { label: 'Prompts', slug: 'guide/prompts' },
                    { label: 'Provenance', slug: 'guide/provenance' },
                    { label: 'Monitoring', slug: 'guide/monitoring' },
                ],
            },
            {
                label: 'Features',
                items: [
                    { label: 'Task Workers', slug: 'features/taskworkers' },
                    { label: 'LLM Integration', slug: 'features/llm-integration' },
                    { label: 'Caching', slug: 'features/caching' },
                    { label: 'Subgraphs', slug: 'features/subgraphs' },
                ],
            },
            {
                label: 'CLI Reference',
                items: [
                    { label: 'Overview', slug: 'cli/overview' },
                    { label: 'Prompt Optimization', slug: 'cli/prompt-optimization' },
                ],
            },
            {
                label: 'API Reference',
                autogenerate: { directory: 'api' },
            },
        ],
		}), sitemap()],
});
