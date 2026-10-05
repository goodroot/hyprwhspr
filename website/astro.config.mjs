// @ts-check
// Single site: marketing at / with docs at /docs/.
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://hyprwhspr.com',
  integrations: [
    starlight({
      title: 'hyprwhspr docs',
      description: 'Canonical hyprwhspr documentation (configuration, installation, benchmarks).',
      disable404Route: true,
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/goodroot/hyprwhspr' }],
      editLink: { baseUrl: 'https://github.com/goodroot/hyprwhspr/edit/main/' },
      customCss: ['@fontsource/jetbrains-mono/latin-400.css', '@fontsource/jetbrains-mono/latin-500.css', '@fontsource/jetbrains-mono/latin-600.css', './src/styles/custom.css'],
      expressiveCode: { themes: ['tokyo-night', 'github-light'] },
      components: { ThemeProvider: './src/components/ThemeProvider.astro' },
      sidebar: [
        { label: 'Guide', items: [{ label: 'Configuration', slug: 'docs/configuration' }, { label: 'Managed installation', slug: 'docs/managed-installation' }] },
        { label: 'Benchmarks', items: [{ autogenerate: { directory: 'docs/benchmarks' } }] },
      ],
    }),
  ],
});
