# website (marketing + docs)

Single Astro site: marketing at `/`, docs at `/docs/` via Starlight.

## Docs pipeline

Canonical Markdown lives in `../docs` (one page per `.md`, no `.mdx`).

```bash
npm run sync # ../docs -> src/content/docs/docs/ + public/docs/
npm test # sync + links suites (temp fixtures, no network)
npm run build # copy install.sh + sync + astro build + link check
npm run dev # sync + astro dev
```

Generated `src/content/docs/docs/*.md` and `public/docs/` are git-ignored; handwritten `src/content/docs/docs/index.mdx` is tracked. CI (`website.yml`) runs test and build on `website/**`, `docs/**`, `scripts/install.sh`; build only, no hosting change.

## Mapping

Canonical `docs/<REL>.md` emits `src/content/docs/docs/<kebab>.md` (per-segment github-slug: lowercase, punctuation stripped, spaces to `-`, then `_` to `-`, nesting preserved). Effective route is `docs/<path>` with one Astro `/index` strip plus one Starlight `/index` strip on the full `docs/` path, then NFC normalize; output files keep generated paths, handwritten `docs/index.mdx` serves `/docs/`. Relative `.md`/`.mdx` links become `/docs/<kebab>/` with index strips and NFC, other relative files become `/docs/<canonical>` mirrored into `public/docs/` resolved from per-page `sourcePath` preserving case and underscores; external, absolute, and hash-only links are untouched.

Link check is `starlight-links-validator` scoped to `/docs/` to avoid marketing anchor quirks; marketing pages are custom pages excluded from docs validation.

## Collisions (all rejected before any writes)

Generated `docs/index.md`, effective `/docs/` (including `INDEX.md`, `INDEX/INDEX.md`, unsluggable `!.md`), duplicate outputs (`GUIDE.md` vs `GUIDE/INDEX.md` or `GUIDE/INDEX/INDEX.md`, NFC variants, Astro-equivalent `A!.md` vs `A.md` and `A B.md` vs `A-B.md`), mirrored assets shadowing route outputs (`guide/index.html` vs `docs/guide`, `index.html` vs landing, ancestor file-vs-dir), case-only asset duplicates, `.mdx` sources, and leading frontmatter are all hard errors.

## Frontmatter

Each page needs a leading `# Title` line; it becomes `title`, the line is stripped, per-page `sourcePath` records the canonical path relative to `docs/` for asset resolution, and per-page `editUrl` points to the canonical source. Frontmatter and `.mdx` are rejected; code fences are never rewritten.
