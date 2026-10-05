// Canonical docs generator for docs-site.
// Reads ../docs recursively (Markdown .md only, one page per source file,
// no topic splits) and writes Starlight-ready Markdown into src/content/docs
// plus static attachments into public/docs. Deterministic, AST-based URL
// rewriting (remark + GFM, so tables/task lists rewrite too).
//
// Mapping:
//   docs/CONFIGURATION.md            -> src/content/docs/configuration.md (route /configuration/)
//   docs/MANAGED_INSTALLATION.md     -> src/content/docs/managed-installation.md
//   docs/benchmarks/<name>.md        -> src/content/docs/benchmarks/<name>.md
// General rule: lowercased canonical relative path for generated files.
// Effective Starlight routes follow the installed Astro/Starlight semantics
// sequentially (not a repeated strip-all loop):
//  1. astro/dist/content/utils.js getContentEntryIdAndSlug:
//     github-slugger per segment, join, strip ONE trailing `/index`
//  2. @astrojs/starlight/dist/utils/routing/index.js normalizeIndexSlug:
//     exact `index` -> `` (root, handwritten index.mdx)
//  3. @astrojs/starlight/dist/utils/slugs.js slugToParam:
//     `index`/`''`/`/` -> root, else strip ONE trailing `/index`, then
//     `.normalize()` (NFC, so composed/decomposed Unicode converge).
// Examples: GUIDE/INDEX.md -> `guide`; GUIDE/INDEX/INDEX.md -> `guide`
// (Astro strips one, Starlight strips one more); GUIDE/INDEX/INDEX/INDEX.md
// -> `guide/index` (only one strip per stage). Generated files keep nested
// `index.md` paths (Astro serves guide/index.md at /guide/), but internal
// links always use effective routes via a separate route map.
// Frontmatter route-override policy: canonical `slug` is rejected with an
// actionable error before any writes (Astro would use it as the content id,
// diverging from rewritten links; only `slug` materially changes the route id
// per installed loader + Starlight schema inspection).
// Non-markdown files (images, benchmark JSON/Python, etc.) are mirrored from
// docs/<rel> to public/docs/<rel> preserving case, and links rewritten to
// <base>/docs/<rel>. Relative .md links are rewritten to <base>/<route>/.
// Fragment-only links (#anchor) and external URLs are preserved verbatim to
// keep canonical heading anchor parity (including duplicates, punctuation,
// emoji). Code fences are never rewritten (AST, not regex).
//
// Markdown-only policy: a canonical .mdx file is rejected with an explicit
// error (never copied as a raw asset). Route collisions (case/underscore-
// hyphen duplicates, effective-route duplicates such as GUIDE.md vs
// GUIDE/INDEX.md or GUIDE.md vs GUIDE/INDEX/INDEX.md (both `guide` after the
// sequential Astro+Starlight index strips), Unicode-normalization duplicates
// such as composed vs decomposed `CAFÉ.md` (both `café` after `.normalize()`),
// or a canonical INDEX.md vs handwritten index.mdx) fail
// before any output writes. Only the initial leading H1 is stripped into
// frontmatter title; a misplaced H1 warns instead of stripping sections.
// Safe canonical frontmatter fields are preserved; generated
// title/description/editUrl always win; canonical `slug` is rejected (never
// preserved) to keep links vs routes in sync. Asset case mismatches resolve to the
// exact on-disk casing (with warning), never a wrong-case URL. Raw HTML
// href/src/srcset outside comments are rewritten; comments untouched.
// All ordering uses codepoint comparison for cross-locale determinism.
//
// Stale outputs are deleted: generated .md files without a canonical source
// and mirrored public/docs files without a canonical counterpart are removed.
// Handwritten src/content/docs/index.mdx is never touched.
//
// Per-page frontmatter editUrl points at the exact canonical file:
//   https://github.com/goodroot/hyprwhspr/edit/main/docs/<canonical file>

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkStringify from 'remark-stringify';
import { visit } from 'unist-util-visit';
import { toString as mdastToString } from 'mdast-util-to-string';
import { normalizeBase, resolveBase } from './site-base.mjs';
import { slug as githubSlug } from 'github-slugger';

// Deterministic codepoint ordering (never locale-dependent).
export function compareCodepoints(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DOCS_SITE_DIR = path.resolve(HERE, '..');
const REPO_ROOT = path.resolve(DOCS_SITE_DIR, '..');
const CANONICAL_ROOT = path.join(REPO_ROOT, 'docs');
const GENERATED_ROOT = path.join(DOCS_SITE_DIR, 'src', 'content', 'docs');
const PUBLIC_DOCS_ROOT = path.join(DOCS_SITE_DIR, 'public', 'docs');

const EDIT_URL_PREFIX = 'https://github.com/goodroot/hyprwhspr/edit/main/docs/';
const CANONICAL_EDIT_PREFIX =
  'https://github.com/goodroot/hyprwhspr/blob/main/docs/';

export const UPSTREAM_EDIT_PREFIX = EDIT_URL_PREFIX;

export function canonicalToGenerated(canonicalRel) {
  // Lowercased kebab-case filename, preserving nested dirs.
  // MANAGED_INSTALLATION.md -> managed-installation.md
  const parts = canonicalRel
    .split('/')
    .map((s) => s.toLowerCase().replace(/_/g, '-'));
  return parts.join('/');
}

export function normalizeCanonicalForLookup(canonicalRel) {
  return canonicalRel.toLowerCase().replace(/_/g, '-');
}

export function generatedToEffectiveRoute(generatedRel) {
  // Effective Starlight route matching the installed Astro/Starlight loader
  // sequentially (mirror exact stages, NOT a strip-all loop):
  // - astro/dist/content/utils.js getContentEntryIdAndSlug:
  //   raw segments -> github-slugger per segment, join, strip ONE trailing `/index`
  // - @astrojs/starlight/dist/utils/routing/index.js normalizeIndexSlug:
  //   exact `index` -> `` (root, handwritten index.mdx)
  // - @astrojs/starlight/dist/utils/slugs.js slugToParam:
  //   `index`/`''`/`/` -> root, else strip ONE trailing `/index`, then `.normalize()`
  // Generated files keep nested `index.md` on disk; links use this route.
  // GUIDE/INDEX/INDEX.md -> `guide`; GUIDE/INDEX/INDEX/INDEX.md -> `guide/index`.
  const withoutExt = generatedRel.replace(/\.mdx?$/i, '');
  const slugged = withoutExt
    .split('/')
    .map((s) => githubSlug(s))
    .join('/');
  let id = slugged.replace(/\/index$/, '');
  if (id === 'index') id = '';
  if (id === 'index' || id === '' || id === '/') return '';
  if (id.endsWith('/index')) id = id.slice(0, -6);
  return id.normalize();
}

export function canonicalToEffectiveRoute(canonicalRel) {
  return generatedToEffectiveRoute(canonicalToGenerated(canonicalRel));
}

export function canonicalToRoute(canonicalRel) {
  // Effective route (not raw filename): nested INDEX.md strips to its dir.
  return canonicalToEffectiveRoute(canonicalRel);
}

export function routeToUrl(route, base) {
  const b = normalizeBase(base ?? resolveBase()) ?? '/hyprwhspr';
  if (b === '/') return `/${route}/`.replace(/\/+/g, '/');
  return `${b}/${route}/`.replace(/\/+/g, '/');
}

export function assetRelToUrl(canonicalAssetRel, base) {
  const b = normalizeBase(base ?? resolveBase()) ?? '/hyprwhspr';
  const rel = canonicalAssetRel.split(path.sep).join('/');
  if (b === '/') return `/docs/${rel}`;
  return `${b}/docs/${rel}`;
}

function isExternalUrl(url) {
  if (!url) return false;
  // scheme:, protocol-relative, data:, mailto:, tel:, etc. Fragment-only handled separately.
  return /^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|\/\/|data:|mailto:|tel:)/.test(url);
}

function splitUrl(url) {
  // Returns { path, query, fragment } where query includes '?', fragment includes '#'.
  let fragment = '';
  let query = '';
  let p = url;
  const hashIdx = p.indexOf('#');
  if (hashIdx !== -1) {
    fragment = p.slice(hashIdx);
    p = p.slice(0, hashIdx);
  }
  const qIdx = p.indexOf('?');
  if (qIdx !== -1) {
    query = p.slice(qIdx);
    p = p.slice(0, qIdx);
  }
  return { path: p, query, fragment };
}

export function rewriteUrl(rawUrl, ctx) {
  // ctx: { currentCanonicalRel, routeMapLower, base, warnings }
  if (rawUrl === undefined || rawUrl === null) return rawUrl;
  const url = String(rawUrl);
  if (url === '') return url;
  if (url.startsWith('#')) return url; // anchor parity: never invent anchors
  if (isExternalUrl(url)) return url;
  if (url.startsWith('/')) return url; // absolute: leave (no canonical absolute links today)
  const { path: urlPath, query, fragment } = splitUrl(url);
  if (urlPath === '') return `${query}${fragment}` || url; // e.g. "?x" edge
  const decodedPath = urlPath;
  // Resolve relative to current canonical file's directory.
  const currentDir = path.posix.dirname(ctx.currentCanonicalRel.split(path.sep).join('/'));
  const joined = path.posix.normalize(
    currentDir === '.' ? decodedPath : path.posix.join(currentDir, decodedPath),
  );
  // Guard against escaping docs root (../..). Preserve + warn.
  if (joined.startsWith('..')) {
    ctx.warnings?.push(`escape: ${ctx.currentCanonicalRel} -> ${url}`);
    return url;
  }
  const lowerJoined = normalizeCanonicalForLookup(joined);
  if (/\.mdx?$/i.test(decodedPath)) {
    const route = ctx.routeMapLower.get(lowerJoined);
    if (route !== undefined) {
      // Standard order: path?query#fragment.
      return `${routeToUrl(route, ctx.base).split('?')[0].split('#')[0]}${query}${fragment}`;
    }
    // Try with fragment/query stripped already handled; if not found, warn + preserve.
    ctx.warnings?.push(`unresolved-md: ${ctx.currentCanonicalRel} -> ${url}`);
    return url;
  }
  // Non-markdown relative asset: check existence under canonical root (case-sensitive).
  // joined preserves original case from the link.
  // Exact-case map (lower -> exact disk rel) resolves case mismatches to the
  // real file instead of emitting a wrong-case URL that 404s on Pages.
  const exactFromMap =
    ctx.existingAssetsLowerMap?.get(lowerJoined) ??
    ctx.existingAssetsLowerMap?.get(joined.toLowerCase());
  let exists = false;
  let resolvedRel = joined;
  try {
    // Sync check via cached set if provided, else filesystem is checked by caller.
    // Here we rely on ctx.existingAssetsExact + lower map sets.
    if (ctx.existingAssetsExact?.has(joined)) {
      exists = true;
      resolvedRel = joined;
    } else if (exactFromMap !== undefined) {
      // Case differs: resolve to the exact on-disk casing, warn loudly.
      exists = true;
      resolvedRel = exactFromMap;
      ctx.warnings?.push(
        `case-mismatch-asset: ${ctx.currentCanonicalRel} -> ${url} (on disk: ${exactFromMap}; link case used: ${joined})`,
      );
    } else if (ctx.existingAssetsLower?.has(lowerJoined)) {
      // Legacy set fallback (tests): still resolvable, but prefer map above.
      exists = true;
    } else {
      // Fallback: will be verified by caller via fs; for pure rewrite (tests),
      // treat as existing if ctx.allowMissingAssets is set.
      if (ctx.allowMissingAssets) {
        exists = true;
      } else {
        // Synchronous existence is not available here; caller pre-populates sets.
        exists = false;
      }
    }
  } catch {
    exists = false;
  }
  if (exists) {
    return `${assetRelToUrl(resolvedRel, ctx.base)}${query}${fragment}`;
  }
  ctx.warnings?.push(`unresolved-asset: ${ctx.currentCanonicalRel} -> ${url}`);
  return url;
}

function rewriteNodeUrl(node, key, ctx) {
  node[key] = rewriteUrl(node[key], ctx);
}

function rewriteSrcsetValue(value, ctx) {
  // srcset: "url [descriptor], url [descriptor], ..." — rewrite each URL only.
  // Descriptors (1x/2x/100w) and commas preserved; externals/fragments preserved
  // by rewriteUrl. Empty entries left untouched.
  const parts = value.split(',');
  let changed = false;
  const out = parts.map((entry) => {
    const m = entry.match(/^(\s*)(\S+)([\s\S]*)$/);
    if (!m) return entry;
    const [, leading, url, rest] = m;
    if (!url) return entry;
    const rewritten = rewriteUrl(url, ctx);
    if (rewritten !== url) changed = true;
    return `${leading}${rewritten}${rest}`;
  });
  void changed;
  return out.join(',');
}

function rewriteAttrsInTagChunk(chunk, ctx) {
  // Only href/src/srcset attributes. Preserve quoting style verbatim.
  return chunk.replace(
    /((?:href|src|srcset)\s*=\s*)(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi,
    (m, prefix, dq, sq, unq, offset, full) => {
      const attrName = prefix.trim().split(/\s/)[0].toLowerCase();
      const original = dq ?? sq ?? unq ?? '';
      const quote = dq !== undefined ? '"' : sq !== undefined ? "'" : '';
      let rewritten;
      if (attrName.startsWith('srcset')) rewritten = rewriteSrcsetValue(original, ctx);
      else rewritten = rewriteUrl(original, ctx);
      if (quote) return `${prefix}${quote}${rewritten}${quote}`;
      return `${prefix}${rewritten}`;
    },
  );
}

function rewriteRawHtmlUrls(html, ctx) {
  // Only href/src/srcset outside HTML comments. Comments, code fences (already
  // excluded: only html nodes reach here), and other attributes preserved verbatim.
  // Handles double quotes, single quotes, unquoted (rare).
  const commentRe = /<!--[\s\S]*?-->/g;
  let last = 0;
  let m;
  let out = '';
  while ((m = commentRe.exec(html)) !== null) {
    out += rewriteAttrsInTagChunk(html.slice(last, m.index), ctx);
    out += m[0]; // comment untouched
    last = m.index + m[0].length;
  }
  out += rewriteAttrsInTagChunk(html.slice(last), ctx);
  return out;
}

export async function transformMarkdown(canonicalRel, rawText, ctx) {
  const parsed = matter(rawText);
  const body = parsed.content;
  const processor = unified().use(remarkParse).use(remarkGfm);
  const tree = processor.parse(body);
  // Extract initial H1 as title; strip only that node (one page per source file).
  let title = undefined;
  const first = tree.children?.[0];
  if (first && first.type === 'heading' && first.depth === 1) {
    title = mdastToString(first).trim();
    tree.children = tree.children.slice(1);
  } else {
    // H1-not-first: do NOT strip arbitrary sections. Warn with actionable
    // explanation so the author can move the H1 to the top.
    const laterH1 = (tree.children ?? []).find((n) => n.type === 'heading' && n.depth === 1);
    if (laterH1) {
      const text = mdastToString(laterH1).trim().slice(0, 80);
      ctx.warnings?.push(
        `misplaced-h1: ${canonicalRel} has H1 "${text}" not as the first block; ` +
          `only a leading H1 is used as the page title. Move the H1 to the top of the file ` +
          `or set frontmatter title explicitly.`,
      );
    }
  }
  if (!title || title === '') {
    if (parsed.data && typeof parsed.data.title === 'string' && parsed.data.title.trim() !== '') {
      title = String(parsed.data.title).trim();
    } else {
      title = path.posix.basename(canonicalRel.replace(/\\/g, '/')).replace(/\.md$/i, '');
    }
  }
  // Description: canonical frontmatter description wins, else first paragraph (160 chars), else title.
  let description = undefined;
  if (parsed.data && typeof parsed.data.description === 'string' && parsed.data.description.trim() !== '') {
    description = String(parsed.data.description).trim();
  } else {
    const firstPara = tree.children?.find((n) => n.type === 'paragraph');
    if (firstPara) {
      const text = mdastToString(firstPara).replace(/\s+/g, ' ').trim();
      if (text) description = text.slice(0, 160);
    }
  }
  if (!description) description = title;
  // Rewrite URLs via AST (links, images, definitions). Code nodes untouched.
  visit(tree, (node) => {
    if (node.type === 'link' || node.type === 'image') {
      rewriteNodeUrl(node, 'url', ctx);
    } else if (node.type === 'definition') {
      rewriteNodeUrl(node, 'url', ctx);
    } else if (node.type === 'html') {
      node.value = rewriteRawHtmlUrls(node.value, ctx);
    }
  });
  const stringifier = unified()
    .use(remarkStringify, {
      bullet: '-',
      fences: true,
      fence: '`',
      listItemIndent: 'one',
      resourceLink: true,
    })
    .use(remarkGfm);
  let markdownBody = stringifier.stringify(tree).trim();
  markdownBody += '\n';
  const canonicalPosix = canonicalRel.split(path.sep).join('/');
  // Preserve safe canonical frontmatter fields (e.g. sidebar, custom keys).
  // Generated title/description/editUrl always win; canonical `slug` is never
  // preserved (defensive strip; generate() rejects it before any writes so
  // links vs Astro routes cannot diverge). Markdown-only policy:
  // canonical sources are .md (see generate() .mdx rejection).
  const { title: _ft, description: _fd, editUrl: _fe, slug: _fs, ...restData } = parsed.data || {};
  void _ft;
  void _fd;
  void _fe;
  void _fs;
  const frontmatterData = {
    title,
    description,
    editUrl: `${EDIT_URL_PREFIX}${canonicalPosix}`,
    ...restData,
  };
  const markdown = matter.stringify(markdownBody, frontmatterData);
  return { title, description, markdown };
}

async function listFilesRecursive(root, exts) {
  const out = [];
  async function walk(dir) {
    let entries = [];
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch (err) {
      if (err?.code === 'ENOENT') return;
      throw err;
    }
    entries.sort((a, b) => compareCodepoints(a.name, b.name));
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) await walk(full);
      else if (e.isFile()) {
        if (!exts || exts.some((ext) => e.name.toLowerCase().endsWith(ext))) out.push(full);
        else if (exts === null) out.push(full);
      }
    }
  }
  await walk(root);
  out.sort(compareCodepoints);
  return out;
}

export async function generate(options = {}) {
  const canonicalRoot = options.canonicalRoot ?? CANONICAL_ROOT;
  const generatedRoot = options.generatedRoot ?? GENERATED_ROOT;
  const publicDocsRoot = options.publicDocsRoot ?? PUBLIC_DOCS_ROOT;
  const base = normalizeBase(options.base ?? resolveBase()) ?? '/hyprwhspr';
  const warnings = [];
  const mdFiles = await listFilesRecursive(canonicalRoot, ['.md']);
  // Canonical sources are Markdown .md only (Markdown-only policy, one page
  // per source file). Reject executable .mdx explicitly: never copy it as a
  // raw asset, never silently ignore it.
  const mdxFiles = await listFilesRecursive(canonicalRoot, ['.mdx']);
  if (mdxFiles.length > 0) {
    const rels = mdxFiles
      .map((f) => path.relative(canonicalRoot, f).split(path.sep).join('/'))
      .sort(compareCodepoints);
    throw new Error(
      `unsupported-format: canonical docs must be Markdown .md only (one page per file); ` +
        `found .mdx source(s): ${rels.join(', ')}. ` +
        `Convert to .md or remove; .mdx is not copied as an asset.`,
    );
  }
  // Build route map (lowercased canonical rel -> route).
  const canonicalRels = [];
  for (const full of mdFiles) {
    const rel = path.relative(canonicalRoot, full).split(path.sep).join('/');
    canonicalRels.push(rel);
  }
  canonicalRels.sort((a, b) => compareCodepoints(a.toLowerCase(), b.toLowerCase()));
  // Collision detection BEFORE ANY output writes: two canonical files must
  // never map to the same generated file/route (case/underscore-hyphen
  // collisions), to the same effective Starlight route (nested INDEX.md
  // strips via the sequential Astro+Starlight stages, e.g. GUIDE.md vs
  // GUIDE/INDEX.md vs GUIDE/INDEX/INDEX.md -> all `guide`; composed vs
  // decomposed Unicode converges via `.normalize()`),
  // and no canonical file may claim the handwritten index slug (root `/`).
  // Effective routes match the installed Astro/Starlight loader sequentially
  // (github-slugger per segment + ONE Astro `/index` strip + root `index`->``
  // + ONE Starlight `/index` strip + `.normalize()`).
  const generatedLowerToCanonical = new Map();
  for (const rel of canonicalRels) {
    const generatedLower = canonicalToGenerated(rel).toLowerCase();
    if (generatedLower === 'index.md') {
      throw new Error(
        `route-collision: canonical ${rel} maps to index.md, which conflicts with the ` +
          `handwritten src/content/docs/index.mdx landing page (same Starlight slug /). ` +
          `Rename the canonical file.`,
      );
    }
    const prev = generatedLowerToCanonical.get(generatedLower);
    if (prev !== undefined) {
      throw new Error(
        `route-collision: canonical files ${prev} and ${rel} both map to generated ` +
          `${canonicalToGenerated(rel)} (case/underscore-hyphen insensitive). ` +
          `Each source file must stay one page with a unique route; rename one file.`,
      );
    }
    generatedLowerToCanonical.set(generatedLower, rel);
  }
  // Route-override policy BEFORE ANY writes: canonical frontmatter must not
  // set `slug`. Astro's glob loader uses frontmatter `slug` as the content id
  // (node_modules/astro/dist/content/loaders/glob.js generateIdDefault), so a
  // surviving `slug` would diverge from filename-derived links and collision
  // checks (e.g. GUIDE.md with `slug: renamed` serves at /renamed/ while links
  // target /guide/). Inspected installed Starlight docsSchema + Astro loader:
  // only `slug` materially changes the route id; sidebar/template/draft/etc.
  // do not. Reject with an actionable error; transformMarkdown also strips
  // `slug` defensively.
  for (const rel of canonicalRels) {
    const full = path.join(canonicalRoot, ...rel.split('/'));
    const raw = await fs.readFile(full, 'utf8');
    const parsed = matter(raw);
    if (parsed.data && Object.hasOwn(parsed.data, 'slug')) {
      throw new Error(
        `route-override: canonical ${rel} sets frontmatter \`slug: ${String(parsed.data.slug)}\`, ` +
          `which would override the filename-derived route and diverge from rewritten links. ` +
          `Remove the \`slug\` field; each source file stays one page at its filename-derived route.`,
      );
    }
  }
  // Effective-route collisions BEFORE ANY writes (Astro would fail with
  // DuplicateContentEntrySlugError at build; catch earlier with a clear error
  // and no partial writes). Generated files keep nested `index.md` paths but
  // links use these effective routes via routeMapLower below.
  const effectiveToCanonical = new Map();
  for (const rel of canonicalRels) {
    const effective = canonicalToEffectiveRoute(rel);
    if (effective === '') {
      throw new Error(
        `route-collision: canonical ${rel} maps to effective Starlight route "/" (root), ` +
          `which conflicts with the handwritten src/content/docs/index.mdx landing page. ` +
          `Rename the canonical file.`,
      );
    }
    const prev = effectiveToCanonical.get(effective);
    if (prev !== undefined) {
      throw new Error(
        `route-collision: canonical files ${prev} and ${rel} both map to effective Starlight route ` +
          `"${effective}" (sequential Astro getContentEntryIdAndSlug ONE \`/index\` strip + ` +
          `Starlight slugToParam ONE \`/index\` strip + \`.normalize()\`; ` +
          `e.g. GUIDE.md vs GUIDE/INDEX.md vs GUIDE/INDEX/INDEX.md all serve at /guide/, ` +
          `composed vs decomposed Unicode converges). ` +
          `Each source file must stay one page with a unique route; rename one file.`,
      );
    }
    effectiveToCanonical.set(effective, rel);
  }
  // Route map (lowercased canonical rel -> effective Starlight route).
  // Output files keep generated paths; links use effective routes.
  const routeMapLower = new Map();
  for (const rel of canonicalRels) {
    routeMapLower.set(normalizeCanonicalForLookup(rel), canonicalToEffectiveRoute(rel));
  }
  // Existing non-md assets for rewrite resolution.
  const allFiles = await listFilesRecursive(canonicalRoot, null);
  const existingAssetsExact = new Set();
  const existingAssetsLower = new Set();
  const existingAssetsLowerMap = new Map();
  const nonMdFiles = [];
  for (const full of allFiles) {
    const rel = path.relative(canonicalRoot, full).split(path.sep).join('/');
    if (/\.mdx?$/i.test(rel)) continue;
    nonMdFiles.push(rel);
    existingAssetsExact.add(rel);
    existingAssetsLower.add(rel.toLowerCase());
    if (!existingAssetsLowerMap.has(rel.toLowerCase())) {
      existingAssetsLowerMap.set(rel.toLowerCase(), rel);
    } else {
      // Two assets differing only by case: ambiguous on case-insensitive
      // checkouts, broken on case-sensitive Pages. Fail loudly.
      const prev = existingAssetsLowerMap.get(rel.toLowerCase());
      if (prev !== rel) {
        throw new Error(
          `asset-case-collision: canonical assets ${prev} and ${rel} differ only by case. ` +
            `Rename one; Pages hosting is case-sensitive.`,
        );
      }
    }
  }
  nonMdFiles.sort(compareCodepoints);
  // Transform each markdown file.
  const expectedGenerated = new Set();
  for (const rel of canonicalRels) {
    const full = path.join(canonicalRoot, ...rel.split('/'));
    const raw = await fs.readFile(full, 'utf8');
    const ctx = {
      currentCanonicalRel: rel,
      routeMapLower,
      base,
      warnings,
      canonicalRoot,
      existingAssetsExact,
      existingAssetsLower,
      existingAssetsLowerMap,
    };
    const { markdown } = await transformMarkdown(rel, raw, ctx);
    const generatedRel = canonicalToGenerated(rel);
    expectedGenerated.add(generatedRel.toLowerCase());
    const outPath = path.join(generatedRoot, ...generatedRel.split('/'));
    await fs.mkdir(path.dirname(outPath), { recursive: true });
    await fs.writeFile(outPath, markdown, 'utf8');
  }
  // Delete stale generated .md files (never touch index.mdx or other handwritten files).
  async function cleanStaleMarkdown(dir, prefixRel) {
    let entries = [];
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch (err) {
      if (err?.code === 'ENOENT') return;
      throw err;
    }
    entries.sort((a, b) => compareCodepoints(a.name, b.name));
    for (const e of entries) {
      const full = path.join(dir, e.name);
      const rel = prefixRel ? `${prefixRel}/${e.name}` : e.name;
      if (e.isDirectory()) {
        await cleanStaleMarkdown(full, rel);
        // Remove empty dirs that are not the root and not handwritten? Only under benchmarks?
        // Keep directories that still contain expected files; remove if empty.
        try {
          const remaining = await fs.readdir(full);
          if (remaining.length === 0) await fs.rmdir(full);
        } catch {}
      } else if (e.isFile()) {
        const lower = e.name.toLowerCase();
        if (lower.endsWith('.md')) {
          if (!expectedGenerated.has(rel.toLowerCase())) {
            await fs.unlink(full);
            warnings.push(`stale-deleted: ${rel}`);
          }
        } else if (lower.endsWith('.mdx')) {
          // Handwritten (index.mdx): never delete.
        }
      }
    }
  }
  await fs.mkdir(generatedRoot, { recursive: true });
  await cleanStaleMarkdown(generatedRoot, '');
  // Mirror non-md assets to public/docs, preserving relative paths.
  await fs.mkdir(publicDocsRoot, { recursive: true });
  for (const rel of nonMdFiles) {
    const src = path.join(canonicalRoot, ...rel.split('/'));
    const dest = path.join(publicDocsRoot, ...rel.split('/'));
    await fs.mkdir(path.dirname(dest), { recursive: true });
    const data = await fs.readFile(src);
    // Avoid rewriting identical files (determinism + mtime stability).
    let existing = null;
    try {
      existing = await fs.readFile(dest);
    } catch {}
    if (!existing || !existing.equals(data)) await fs.writeFile(dest, data);
  }
  // Delete stale mirrored assets.
  async function cleanStalePublic(dir, prefixRel, expectedSet) {
    let entries = [];
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch (err) {
      if (err?.code === 'ENOENT') return;
      throw err;
    }
    entries.sort((a, b) => compareCodepoints(a.name, b.name));
    for (const e of entries) {
      const full = path.join(dir, e.name);
      const rel = prefixRel ? `${prefixRel}/${e.name}` : e.name;
      if (e.isDirectory()) {
        await cleanStalePublic(full, rel, expectedSet);
        try {
          const remaining = await fs.readdir(full);
          if (remaining.length === 0) await fs.rmdir(full);
        } catch {}
      } else if (e.isFile()) {
        if (!expectedSet.has(rel)) {
          await fs.unlink(full);
          warnings.push(`stale-asset-deleted: docs/${rel}`);
        }
      }
    }
  }
  await cleanStalePublic(publicDocsRoot, '', new Set(nonMdFiles));
  return {
    canonicalRels,
    expectedGenerated: [...expectedGenerated].sort(compareCodepoints),
    nonMdFiles,
    warnings,
    base,
  };
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  generate()
    .then((result) => {
      console.log(
        `docs-site: generated ${result.canonicalRels.length} pages, mirrored ${result.nonMdFiles.length} assets (base ${result.base})`,
      );
      for (const w of result.warnings) console.warn(`docs-site warning: ${w}`);
    })
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}

export { CANONICAL_EDIT_PREFIX, CANONICAL_ROOT, GENERATED_ROOT, PUBLIC_DOCS_ROOT };
