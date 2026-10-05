# hyprwhspr docs site (isolated Starlight)

Isolated [Astro Starlight](https://starlight.astro.build/) site that renders the canonical Markdown under `docs/` verbatim — one page per source file, no topic splits. No runtime application code, no `website/` or `hyprwhspr.com` changes.

## Canonical source pipeline

- Canonical root: `../docs` (recursive `*.md`, Markdown-only). A canonical `.mdx` file fails the build with an explicit `unsupported-format` error — it is never copied as a raw asset. Each source file stays one page, no topic splits.
- Generated pages: `src/content/docs/` (lowercased paths, e.g. `CONFIGURATION.md` → `configuration.md`, `benchmarks/*.md` → `benchmarks/*.md`). Only the initial leading `# H1` is stripped into frontmatter `title`; a later (misplaced) H1 is kept and warns with an actionable message instead of stripping sections. `H2/H3/H4` hierarchy, duplicate headings, punctuation, and emoji are preserved for anchor parity.
- Frontmatter adaptation: safe canonical fields (e.g. `sidebar`, custom keys) are preserved; generated `title`/`description`/`editUrl` always win. Canonical `description` feeds the generated description when present, otherwise the first paragraph (160 chars) or title is used. Canonical `slug` is rejected with an actionable `route-override` error before any writes (Astro would use it as the content id, diverging from rewritten links; only `slug` materially changes the route per installed loader + Starlight schema inspection).
- Route safety: two canonical files mapping to the same generated file (case/underscore-hyphen collisions such as `FOO_BAR.md` vs `FOO-BAR.md`) or to the same effective Starlight route (e.g. `GUIDE.md` vs `GUIDE/INDEX.md` vs `GUIDE/INDEX/INDEX.md`, all `guide` after the sequential Astro + Starlight single-`/index` strips; composed vs decomposed `CAFÉ.md` both `café` after `.normalize()`) fail before any output writes, as does a canonical `INDEX.md`/`index.md` claiming the handwritten landing slug (`/`). Effective routes follow the installed Astro/Starlight semantics sequentially (github-slugger per segment + one Astro `/index` strip + root `index`→`` + one Starlight `/index` strip + `.normalize()`); generated files keep nested `index.md` paths but links use effective routes.
- URL rewriting uses a Markdown AST (`remark` + GFM, so tables/task lists rewrite too), never regex over raw text, so fenced code and string examples are untouched. Relative `.md` links become `<base>/<effective-route>/` (e.g. `GUIDE/INDEX.md` → `<base>/guide/`; query `?` and `#fragment` preserved), images and benchmark `JSON/Python` attachments become `<base>/docs/<path>`, fragment-only `#anchors` and external URLs are preserved. Asset links resolve to the exact on-disk casing (case mismatches warn); unresolvable links warn instead of inventing anchors. Raw HTML `href`/`src`/`srcset` outside comments are rewritten; comments untouched; `../..` escapes warn and are preserved.
- Non-markdown files are mirrored `docs/<rel>` → `public/docs/<rel>` (e.g. `docs/assets/pill-states.png`, benchmark receipt/runner). Stale generated `.md` and stale mirrored assets are deleted. Handwritten `src/content/docs/index.mdx` (navigation only) is never overwritten.
- Per-page `editUrl` points at the exact canonical file: `https://github.com/goodroot/hyprwhspr/edit/main/docs/<file>`.
- Sidebar: `Guide` (configuration, managed installation) + auto-generated `Benchmarks`. On-page TOC and Pagefind search are Starlight defaults. Styling is plain Starlight variables in `src/styles/custom.css` (no external images).
- Ordering is codepoint-based (never locale-dependent), so output is byte-deterministic across environments.

Regenerate deterministically:

```bash
npm run generate
```

## Requirements

Node `>=22.12` (see `engines`). CI uses Node 24 with `npm ci` and the committed `package-lock.json`.

- Astro `7.3.5` + Starlight `0.42.5` (compatible per npm peer metadata: Starlight `0.42.x` requires `astro ^7.2.10`).

## Install / build / dev / preview

From `docs-site/`:

```bash
npm ci
npm test            # generator + verifier suites (temp fixtures, no network)
npm run build       # generate + astro build
npm run verify      # offline check of dist href/src, fragments, every mirrored asset, pagefind
npm run dev         # generate + astro dev
npm run preview     # astro preview (serve dist)
```

Generated content (`src/content/docs/*.md`, `src/content/docs/**/*.md` including future nested dirs, `public/docs/`, `dist/`, `.astro/`) is git-ignored; never commit it. The verifier derives the expected `dist/docs/*` asset list from the canonical tree (not a hardcoded sample) and treats absolute URLs missing the project `base` as errors.

## Hosting, Pages opt-in, and site/base overrides

Defaults target GitHub Project Pages derived from `GITHUB_REPOSITORY` (fallback `goodroot/hyprwhspr`):

- `site` → `https://<owner>.github.io`
- `base` → `/hyprwhspr`

Explicit overrides for custom hosting or root preview:

```bash
DOCS_SITE_URL=https://docs.example.com DOCS_BASE_PATH=/ npm run build
DOCS_SITE_URL=https://example.com DOCS_BASE_PATH=/ npm run build   # root preview
```

Aliases `DOCS_SITE` and `DOCS_BASE` are also accepted. The `docs-site.yml` workflow runs `actions/configure-pages` (with `enablement: false`, never enabling Pages) only when deployment is eligible — main push/dispatch plus `DOCS_PAGES_ENABLED == 'true'` — and passes its `origin`/`base_path` through these variables. All other builds (PRs, forks, local) skip the Pages API entirely and use `GITHUB_REPOSITORY`-derived defaults or the explicit `DOCS_*` overrides above.

Existing Pages caveat: this repo may already serve `website/` (hyprwhspr.com) via Pages. Deploying `docs-site/dist` to Pages would replace that deployment. Deployment is therefore opt-in only: push to `main` (or manual dispatch) **and** the repository variable `DOCS_PAGES_ENABLED == 'true'`. Otherwise the workflow only builds, tests, and verifies. No `CNAME` or DNS changes are made here.
