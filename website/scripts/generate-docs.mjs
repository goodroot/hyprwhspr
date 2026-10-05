// Canonical docs generator: ../docs .md to Starlight Markdown under docs/.
// Deterministic AST rewriting; one page per source file.
// See README for mapping, collisions, and frontmatter policy.

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
const WEBSITE_DIR = path.resolve(HERE, '..');
const REPO_ROOT = path.resolve(WEBSITE_DIR, '..');
const CANONICAL_ROOT = path.join(REPO_ROOT, 'docs');
const GENERATED_ROOT = path.join(WEBSITE_DIR, 'src', 'content', 'docs');
const PUBLIC_DOCS_ROOT = path.join(WEBSITE_DIR, 'public', 'docs');

const EDIT_URL_PREFIX = 'https://github.com/goodroot/hyprwhspr/edit/main/docs/';
const CANONICAL_EDIT_PREFIX =
  'https://github.com/goodroot/hyprwhspr/blob/main/docs/';

export const UPSTREAM_EDIT_PREFIX = EDIT_URL_PREFIX;

export function canonicalToGenerated(canonicalRel) {
  const parts = canonicalRel.split('/').map((s) => s.toLowerCase().replace(/_/g, '-'));
  return `docs/${parts.join('/')}`;
}

export function normalizeCanonicalForLookup(canonicalRel) {
  return canonicalRel.toLowerCase().replace(/_/g, '-');
}

export function generatedToEffectiveRoute(generatedRel) {
  // Effective route per installed Astro+Starlight index stages on full path.
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
  return canonicalToEffectiveRoute(canonicalRel);
}

export function effectiveRouteToPathname(effective) {
  // Mirrors stringifyParams trim + sanitize for spread route.
  if (effective === '') return '/';
  const trimmed = effective.replace(/^\/|\/$/g, '');
  const sanitized = trimmed.normalize().replace(/#/g, '%23').replace(/\?/g, '%3F');
  return sanitized ? `/${sanitized}` : '/';
}

export function pathnameToOutputRel(pathname) {
  // Directory build output (config sets no format override).
  if (pathname === '/' || pathname === '') return 'index.html';
  const noTrailing = pathname.endsWith('/') ? pathname.slice(0, -1) : pathname;
  const stripped = noTrailing.startsWith('/') ? noTrailing.slice(1) : noTrailing;
  return `${stripped}/index.html`;
}

export function normalizeOutputRel(rel) {
  // Lowercase + collapse slashes for alias-safe comparison.
  return rel.normalize().replace(/\/+/g, '/').toLowerCase();
}

export function routeToUrl(route, base) {
  const b = normalizeBase(base ?? resolveBase()) ?? '/';
  if (b === '/') return `/${route}/`.replace(/\/+/g, '/');
  return `${b}/${route}/`.replace(/\/+/g, '/');
}

export function assetRelToUrl(canonicalAssetRel, base) {
  const b = normalizeBase(base ?? resolveBase()) ?? '/';
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
  if (rawUrl === undefined || rawUrl === null) return rawUrl;
  const url = String(rawUrl);
  if (url === '') return url;
  if (url.startsWith('#')) return url; // anchor parity: never invent anchors
  if (isExternalUrl(url)) return url;
  if (url.startsWith('/')) return url; // absolute: leave (no canonical absolute links today)
  const { path: urlPath, query, fragment } = splitUrl(url);
  if (urlPath === '') return `${query}${fragment}` || url;
  const decodedPath = urlPath;
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
      return `${routeToUrl(route, ctx.base).split('?')[0].split('#')[0]}${query}${fragment}`;
    }
    ctx.warnings?.push(`unresolved-md: ${ctx.currentCanonicalRel} -> ${url}`);
    return url;
  }
  // Non-markdown asset: resolve exact on-disk casing via lower map.
  const exactFromMap =
    ctx.existingAssetsLowerMap?.get(lowerJoined) ??
    ctx.existingAssetsLowerMap?.get(joined.toLowerCase());
  let exists = false;
  let resolvedRel = joined;
  try {
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
      if (ctx.allowMissingAssets) {
        exists = true;
      } else {
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
  // Rewrite href/src/srcset outside HTML comments only.
  const commentRe = /<!--[\s\S]*?-->/g;
  let last = 0;
  let m;
  let out = '';
  while ((m = commentRe.exec(html)) !== null) {
    out += rewriteAttrsInTagChunk(html.slice(last, m.index), ctx);
    out += m[0];
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
    // H1-not-first: do NOT strip arbitrary sections. Warn with actionable explanation so the author can move the H1 to the top.
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
  // Preserve safe frontmatter; generated title/description/editUrl win, slug stripped.
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
  const base = normalizeBase(options.base ?? resolveBase()) ?? '/';
  const warnings = [];
  const mdFiles = await listFilesRecursive(canonicalRoot, ['.md']);
  // Canonical sources are Markdown .md only; reject .mdx explicitly.
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
  const canonicalRels = [];
  for (const full of mdFiles) {
    const rel = path.relative(canonicalRoot, full).split(path.sep).join('/');
    canonicalRels.push(rel);
  }
  canonicalRels.sort((a, b) => compareCodepoints(a.toLowerCase(), b.toLowerCase()));
  // Collision checks run before any writes.
  const generatedLowerToCanonical = new Map();
  for (const rel of canonicalRels) {
    const generatedLower = canonicalToGenerated(rel).toLowerCase();
    if (generatedLower === 'docs/index.md') {
      throw new Error(
        `route-collision: canonical ${rel} maps to docs/index.md, which conflicts with the ` +
          `handwritten src/content/docs/docs/index.mdx landing page (route /docs/). ` +
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
  // Reject canonical frontmatter slug before any writes.
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
  // Reject effective-route collisions before any writes.
  const outputToCanonical = new Map();
  const landingNorm = normalizeOutputRel('docs/index.html');
  for (const rel of canonicalRels) {
    const effective = canonicalToEffectiveRoute(rel);
    const outputNorm = normalizeOutputRel(pathnameToOutputRel(effectiveRouteToPathname(effective)));
    if (outputNorm === landingNorm || outputNorm === 'index.html') {
      throw new Error(
        `route-collision: canonical ${rel} maps to effective Starlight route "/docs/", ` +
          `which conflicts with the handwritten src/content/docs/docs/index.mdx landing page. ` +
          `Rename the canonical file.`,
      );
    }
    if (effective.includes('//') || effective.startsWith('/') || effective.endsWith('/')) {
      throw new Error(
        `route-collision: canonical ${rel} maps to degenerate route "${effective}" ` +
          `(empty segment from unsluggable filename); rename the canonical file.`,
      );
    }
    const prev = outputToCanonical.get(outputNorm);
    if (prev !== undefined) {
      throw new Error(
        `route-collision: canonical files ${prev} and ${rel} both map to effective Starlight route ` +
          `"${effective}" (sequential Astro getContentEntryIdAndSlug ONE \`/index\` strip + ` +
          `Starlight slugToParam ONE \`/index\` strip + \`.normalize()\`; ` +
          `e.g. GUIDE.md vs GUIDE/INDEX.md vs GUIDE/INDEX/INDEX.md all serve at /docs/guide/, ` +
          `composed vs decomposed Unicode converges). ` +
          `Each source file must stay one page with a unique route; rename one file.`,
      );
    }
    outputToCanonical.set(outputNorm, rel);
  }
  // Route map uses effective routes; output files keep generated paths.
  const routeMapLower = new Map();
  for (const rel of canonicalRels) {
    routeMapLower.set(normalizeCanonicalForLookup(rel), canonicalToEffectiveRoute(rel));
  }
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
      // Same lowercased assets with different case fail loudly.
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
  const routeNorms = [...outputToCanonical.keys(), landingNorm];
  const routeNormToRel = new Map([...outputToCanonical.entries(), [landingNorm, 'handwritten src/content/docs/docs/index.mdx']]);
  for (const assetRel of nonMdFiles) {
    const assetNorm = normalizeOutputRel(`docs/${assetRel}`);
    const exact = routeNormToRel.get(assetNorm);
    if (exact !== undefined) {
      throw new Error(
        `route-collision: mirrored asset "docs/${assetRel}" collides with route output "${assetNorm}" ` +
          `(from ${exact}); rename one file.`,
      );
    }
    for (const routeNorm of routeNorms) {
      if (routeNorm.startsWith(`${assetNorm}/`) || assetNorm.startsWith(`${routeNorm}/`)) {
        const owner = routeNormToRel.get(routeNorm) ?? routeNorm;
        throw new Error(
          `route-collision: mirrored asset "docs/${assetRel}" blocks route output "${routeNorm}" ` +
            `(from ${owner}); rename one file.`,
        );
      }
    }
  }
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
        }
      }
    }
  }
  await fs.mkdir(generatedRoot, { recursive: true });
  await cleanStaleMarkdown(generatedRoot, '');
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
        `website: generated ${result.canonicalRels.length} pages, mirrored ${result.nonMdFiles.length} assets (base ${result.base})`,
      );
      for (const w of result.warnings) console.warn(`website warning: ${w}`);
    })
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}

export { CANONICAL_EDIT_PREFIX, CANONICAL_ROOT, GENERATED_ROOT, PUBLIC_DOCS_ROOT };
