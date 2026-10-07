// @ts-check
// Single site: marketing at / with docs at /docs/.
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import { satteri } from '@astrojs/markdown-satteri';
import { docsLinks } from './src/docs-links.mjs';
import { sidebar } from './src/sidebar.mjs';

export default defineConfig({
  site: 'https://hyprwhspr.com',
  markdown: { processor: satteri({ mdastPlugins: [docsLinks] }) },
  integrations: [
    starlight({
      title: 'hyprwhspr docs',
      description: 'Configure, install and benchmark hyprwhspr.',
      disable404Route: true,
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/goodroot/hyprwhspr' }],
      editLink: { baseUrl: 'https://github.com/goodroot/hyprwhspr/edit/main/website/' },
      customCss: ['@fontsource/jetbrains-mono/latin-400.css', '@fontsource/jetbrains-mono/latin-500.css', '@fontsource/jetbrains-mono/latin-600.css', './src/styles/custom.css'],
      expressiveCode: { themes: ['tokyo-night', 'github-light'] },
      components: { ThemeProvider: './src/components/ThemeProvider.astro' },
      sidebar,
      plugins: [starlightLinksValidator({ exclude: ({ link }) => link.startsWith('/') && !link.startsWith('/docs/') })],
    }),
  ],
});
