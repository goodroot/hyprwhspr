// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import { satteri } from '@astrojs/markdown-satteri';
import { docsLinks } from './src/docs-links.mjs';

// Pages come from ../docs via scripts/sync-docs.mjs; the sidebar autogenerates.
export default defineConfig({
  site: 'https://docs.hyprwhspr.com',
  markdown: { processor: satteri({ mdastPlugins: [docsLinks] }) },
  integrations: [
    starlight({
      title: 'hyprwhspr docs',
      description: 'Configure, install and benchmark hyprwhspr.',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/goodroot/hyprwhspr' },
      ],
      editLink: { baseUrl: 'https://github.com/goodroot/hyprwhspr/edit/main/docs-site/' },
      customCss: [
        '@fontsource/jetbrains-mono/latin-400.css',
        '@fontsource/jetbrains-mono/latin-500.css',
        '@fontsource/jetbrains-mono/latin-600.css',
        './src/styles/custom.css',
      ],
      expressiveCode: { themes: ['tokyo-night', 'github-light'] },
      components: { ThemeProvider: './src/components/ThemeProvider.astro' },
      plugins: [starlightLinksValidator()],
    }),
  ],
});
