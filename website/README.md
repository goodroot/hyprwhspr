# website

hyprwhspr.com: marketing pages at `/`, Starlight docs at `/docs/`. Hosted on Netlify (`netlify.toml`), base directory `website`.

```bash
npm ci
npm run dev     # sync docs + live preview
npm run build   # copy install.sh + sync docs + build + docs link check -> dist/
```

Docs come from `../docs`; edit there. Each `.md` becomes a page titled by its leading `# H1`, relative `.md` links become routes, and other files are served from `/docs/`. Synced pages and `public/docs/` are gitignored.
