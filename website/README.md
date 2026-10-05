# website (marketing + docs)

Single Astro site: marketing at `/`, docs at `/docs/` via Starlight.

## Docs pipeline

Canonical Markdown lives in `../docs` (one page per `.md`, no `.mdx`).

```bash
npm run generate # ../docs -> src/content/docs/docs/ + public/docs/
npm test # generator + verifier suites (temp fixtures, no network)
npm run build # copy install.sh + generate + astro build
npm run verify # check dist links, assets, marketing, install.sh, pagefind
npm run dev # generate + astro dev
```

Generated `src/content/docs/docs/*.md` and `public/docs/` are git-ignored; handwritten `src/content/docs/docs/index.mdx` is tracked. CI (`website.yml`) runs test, build, verify on `website/**`, `docs/**`, `scripts/install.sh`; build only, no hosting change.

## Mapping

Canonical `docs/<REL>.md` emits generated `src/content/docs/docs/<kebab>.md` (lowercase, `_` to `-`, nesting preserved). Links use the effective Starlight route `docs/<route>`: per-segment `github-slugger`, one Astro `/index` strip plus one Starlight `/index` strip on the full `docs/` path, then NFC normalize. Output files keep generated paths; the handwritten `docs/index.mdx` serves `/docs/`.

## Collisions (all rejected before any writes)

Generated `docs/index.md`, effective `/docs/` (including `INDEX.md`, `INDEX/INDEX.md`, unsluggable `!.md`, degenerate `//` or trailing-slash routes), duplicate outputs (`GUIDE.md` vs `GUIDE/INDEX.md`, triple-`INDEX`, NFC `CAFE` variants), mirrored assets shadowing route outputs (`guide/index.html` vs `docs/guide`, `index.html` vs landing, ancestor-blocking files like `guide` vs `docs/guide/`), case-only asset duplicates, `.mdx` sources, and canonical frontmatter `slug` are all hard errors.

## Frontmatter

Generated `title` (leading H1, else frontmatter `title`, else filename), `description` (frontmatter, else first paragraph truncated to 160 chars, else title), and per-page `editUrl` to the canonical source always win; `slug` is stripped and never survives. Other safe fields (e.g. `sidebar`) are preserved.
