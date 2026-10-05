// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { resolveBase, resolveSite } from './scripts/site-base.mjs';

// Isolated docs site. Canonical Markdown lives in ../docs and is generated
// into src/content/docs at build/dev time (see scripts/generate-docs.mjs).
// Never enable GitHub Pages automatically; deployment is opt-in via the
// DOCS_PAGES_ENABLED repository variable in .github/workflows/docs-site.yml.
// No CNAME or DNS changes. website/ and hyprwhspr.com are untouched.

const site = resolveSite();
const base = resolveBase();

export default defineConfig({
  site,
  base,
  integrations: [
    starlight({
      title: 'hyprwhspr docs',
      description: 'Canonical hyprwhspr documentation (configuration, installation, benchmarks).',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/goodroot/hyprwhspr' },
      ],
      editLink: {
        // Fallback for handwritten pages. Generated pages set a per-page
        // editUrl pointing at the exact canonical file under docs/.
        baseUrl: 'https://github.com/goodroot/hyprwhspr/edit/main/',
      },
      customCss: [
        '@fontsource/jetbrains-mono/latin-400.css',
        '@fontsource/jetbrains-mono/latin-500.css',
        '@fontsource/jetbrains-mono/latin-600.css',
        './src/styles/custom.css',
      ],
      expressiveCode: {
        // Tokyo Night for dark (exact marketing background #1a1b26),
        // GitHub Light for an accessible light mode. Starlight keeps the
        // active block in sync with the site theme automatically.
        themes: ['tokyo-night', 'github-light'],
      },
      components: {
        // First-visit default is dark (marketing match); explicit
        // light/dark/Auto choices are respected. See component.
        ThemeProvider: './src/components/ThemeProvider.astro',
      },
      sidebar: [
        {
          label: 'Guide',
          items: [
            { label: 'Configuration', slug: 'configuration' },
            { label: 'Managed installation', slug: 'managed-installation' },
          ],
        },
        {
          label: 'Benchmarks',
          items: [{ autogenerate: { directory: 'benchmarks' } }],
        },
      ],
    }),
  ],
});
