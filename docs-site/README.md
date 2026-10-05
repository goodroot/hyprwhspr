# docs-site

Starlight build of `../docs`. Edit `docs/`; pages and assets here are synced at build time and gitignored.

```bash
npm ci
npm run dev     # sync + live preview
npm run build   # sync + build + link check -> dist/
```

Each `docs/*.md` becomes a page titled by its leading `# H1`. Relative links to `.md` files become routes; other files are served from `/docs/`.

Hosted on Netlify (`netlify.toml`), base directory `docs-site`.
