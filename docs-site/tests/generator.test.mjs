// Generator tests: temp fixtures, no network, no repo mutation.
// Covers frontmatter/title escaping, code-fence preservation, link/ref/image/
// raw-HTML rewriting, nested paths, anchor parity (duplicates/punctuation/
// emoji), assets/attachments, stale deletion, determinism, site/base variations.

import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import { visit } from 'unist-util-visit';
import { toString as mdastToString } from 'mdast-util-to-string';
import {
  canonicalToGenerated,
  canonicalToEffectiveRoute,
  canonicalToRoute,
  compareCodepoints,
  generatedToEffectiveRoute,
  routeToUrl,
  assetRelToUrl,
  rewriteUrl,
  transformMarkdown,
  generate,
} from '../scripts/generate-docs.mjs';
import { normalizeBase, resolveBase, resolveSite } from '../scripts/site-base.mjs';

function makeCtx(overrides = {}) {
  return {
    currentCanonicalRel: 'CONFIGURATION.md',
    routeMapLower: new Map([
      ['configuration.md', 'configuration'],
      ['managed-installation.md', 'managed-installation'],
      ['benchmarks/orukeet-linux-20260918.md', 'benchmarks/orukeet-linux-20260918'],
      ['a/b/c.md', 'a/b/c'],
    ]),
    base: '/hyprwhspr',
    warnings: [],
    existingAssetsExact: new Set(['assets/pill-states.png', 'benchmarks/orukeet-linux-20260918.json']),
    existingAssetsLower: new Set(['assets/pill-states.png', 'benchmarks/orukeet-linux-20260918.json']),
    allowMissingAssets: true,
    ...overrides,
  };
}

describe('canonical mapping', () => {
  it('lowercases generated filenames and routes, preserves nesting', () => {
    assert.equal(canonicalToGenerated('CONFIGURATION.md'), 'configuration.md');
    assert.equal(canonicalToGenerated('MANAGED_INSTALLATION.md'), 'managed-installation.md');
    assert.equal(
      canonicalToGenerated('benchmarks/orukeet-linux-20260918.md'),
      'benchmarks/orukeet-linux-20260918.md',
    );
    assert.equal(canonicalToGenerated('A/B/C.MD'), 'a/b/c.md');
    assert.equal(canonicalToRoute('CONFIGURATION.md'), 'configuration');
    assert.equal(canonicalToRoute('MANAGED_INSTALLATION.md'), 'managed-installation');
    assert.equal(canonicalToRoute('benchmarks/orukeet-linux-20260918.md'), 'benchmarks/orukeet-linux-20260918');
  });

  it('routeToUrl respects base variations', () => {
    assert.equal(routeToUrl('configuration', '/hyprwhspr'), '/hyprwhspr/configuration/');
    assert.equal(routeToUrl('configuration', '/'), '/configuration/');
    assert.equal(routeToUrl('configuration', '/custom/'), '/custom/configuration/');
    assert.equal(routeToUrl('a/b/c', '/hyprwhspr'), '/hyprwhspr/a/b/c/');
  });

  it('assetRelToUrl respects base variations', () => {
    assert.equal(assetRelToUrl('assets/pill-states.png', '/hyprwhspr'), '/hyprwhspr/docs/assets/pill-states.png');
    assert.equal(assetRelToUrl('assets/pill-states.png', '/'), '/docs/assets/pill-states.png');
    assert.equal(
      assetRelToUrl('benchmarks/orukeet-linux-20260918.json', '/custom'),
      '/custom/docs/benchmarks/orukeet-linux-20260918.json',
    );
  });
});

describe('site/base resolution', () => {
  const OLD = { ...process.env };
  afterEach(() => {
    process.env = { ...OLD };
  });

  it('defaults to upstream owner/repo project pages', () => {
    delete process.env.DOCS_SITE_URL;
    delete process.env.DOCS_SITE;
    delete process.env.DOCS_BASE_PATH;
    delete process.env.DOCS_BASE;
    delete process.env.GITHUB_REPOSITORY;
    assert.equal(resolveSite(), 'https://goodroot.github.io');
    assert.equal(resolveBase(), '/hyprwhspr');
  });

  it('derives owner from GITHUB_REPOSITORY', () => {
    delete process.env.DOCS_SITE_URL;
    delete process.env.DOCS_SITE;
    process.env.GITHUB_REPOSITORY = 'octo/example';
    assert.equal(resolveSite(), 'https://octo.github.io');
  });

  it('supports explicit overrides and aliases', () => {
    process.env.DOCS_SITE_URL = 'https://example.com/docs/';
    process.env.DOCS_BASE_PATH = '/custom/';
    assert.equal(resolveSite(), 'https://example.com/docs');
    assert.equal(resolveBase(), '/custom');
    delete process.env.DOCS_SITE_URL;
    process.env.DOCS_SITE = 'https://alias.example';
    delete process.env.DOCS_BASE_PATH;
    process.env.DOCS_BASE = 'root';
    assert.equal(resolveSite(), 'https://alias.example');
    assert.equal(resolveBase(), '/root');
  });

  it('normalizes root base for preview', () => {
    assert.equal(normalizeBase('/'), '/');
    assert.equal(normalizeBase(''), '/');
    assert.equal(normalizeBase('/hyprwhspr/'), '/hyprwhspr');
    assert.equal(normalizeBase('hyprwhspr'), '/hyprwhspr');
  });
});

describe('rewriteUrl', () => {
  it('preserves fragments, externals, absolutes', () => {
    const ctx = makeCtx();
    assert.equal(rewriteUrl('#minimal-configuration', ctx), '#minimal-configuration');
    assert.equal(rewriteUrl('#setup-1', ctx), '#setup-1');
    assert.equal(rewriteUrl('https://example.com/x.md', ctx), 'https://example.com/x.md');
    assert.equal(rewriteUrl('mailto:a@b.c', ctx), 'mailto:a@b.c');
    assert.equal(rewriteUrl('/absolute/path', ctx), '/absolute/path');
  });

  it('rewrites relative .md to routes with base + preserves fragment/query', () => {
    const ctx = makeCtx({ currentCanonicalRel: 'CONFIGURATION.md' });
    assert.equal(
      rewriteUrl('benchmarks/orukeet-linux-20260918.md', ctx),
      '/hyprwhspr/benchmarks/orukeet-linux-20260918/',
    );
    assert.equal(
      rewriteUrl('MANAGED_INSTALLATION.md', ctx),
      '/hyprwhspr/managed-installation/',
    );
    // Fragment preserved verbatim (anchor parity).
    assert.equal(
      rewriteUrl('benchmarks/orukeet-linux-20260918.md#results', ctx),
      '/hyprwhspr/benchmarks/orukeet-linux-20260918/#results',
    );
    const rootCtx = makeCtx({ currentCanonicalRel: 'benchmarks/orukeet-linux-20260918.md' });
    // Nested file linking upward (case-insensitive .MD handling via lower map).
    rootCtx.routeMapLower.set('configuration.md', 'configuration');
    assert.equal(rewriteUrl('../CONFIGURATION.md#backends', rootCtx), '/hyprwhspr/configuration/#backends');
  });

  it('rewrites relative assets to static files with base', () => {
    const ctx = makeCtx({ currentCanonicalRel: 'CONFIGURATION.md' });
    assert.equal(
      rewriteUrl('assets/pill-states.png', ctx),
      '/hyprwhspr/docs/assets/pill-states.png',
    );
    const bench = makeCtx({ currentCanonicalRel: 'benchmarks/orukeet-linux-20260918.md' });
    assert.equal(
      rewriteUrl('orukeet-linux-20260918.json', bench),
      '/hyprwhspr/docs/benchmarks/orukeet-linux-20260918.json',
    );
  });

  it('warns and preserves unresolvable links instead of inventing anchors', () => {
    const ctx = makeCtx({ currentCanonicalRel: 'CONFIGURATION.md', allowMissingAssets: false });
    ctx.existingAssetsExact = new Set();
    ctx.existingAssetsLower = new Set();
    const out = rewriteUrl('missing/file.md', ctx);
    assert.equal(out, 'missing/file.md');
    assert.ok(ctx.warnings.some((w) => w.includes('unresolved-md')));
    const out2 = rewriteUrl('assets/nope.png', ctx);
    assert.equal(out2, 'assets/nope.png');
    assert.ok(ctx.warnings.some((w) => w.includes('unresolved-asset')));
  });

  it('respects root base variation', () => {
    const ctx = makeCtx({ base: '/' });
    assert.equal(
      rewriteUrl('benchmarks/orukeet-linux-20260918.md', ctx),
      '/benchmarks/orukeet-linux-20260918/',
    );
    assert.equal(rewriteUrl('assets/pill-states.png', ctx), '/docs/assets/pill-states.png');
  });
});

describe('transformMarkdown', () => {
  it('strips only initial H1 into frontmatter title with safe escaping', async () => {
    const ctx = makeCtx();
    const raw = '# Foo "bar": baz & qux\n\nBody.\n';
    const { title, markdown } = await transformMarkdown('CONFIGURATION.md', raw, ctx);
    assert.equal(title, 'Foo "bar": baz & qux');
    const fm = matter(markdown);
    assert.equal(fm.data.title, 'Foo "bar": baz & qux');
    assert.ok(fm.data.editUrl.endsWith('/docs/CONFIGURATION.md'));
    // H1 gone, body retained.
    assert.ok(!fm.content.startsWith('# Foo'));
    assert.ok(fm.content.includes('Body.'));
  });

  it('retains H2/H3/H4 hierarchy and duplicate/punctuation/emoji headings verbatim', async () => {
    const ctx = makeCtx();
    const raw = [
      '# Top',
      '',
      '## Setup',
      '',
      '### Setup',
      '',
      '#### Cohere 🇨🇦',
      '',
      '#### Available models',
      '',
      '#### Available models',
      '',
    ].join('\n');
    const { markdown } = await transformMarkdown('X.md', raw, ctx);
    const fm = matter(markdown);
    assert.ok(fm.content.includes('## Setup'));
    assert.ok(fm.content.includes('### Setup'));
    assert.ok(fm.content.includes('#### Cohere 🇨🇦'));
    // Duplicates preserved (parity, not deduplicated).
    assert.equal((fm.content.match(/#### Available models/g) || []).length, 2);
  });

  it('does not rewrite markdown link syntax inside fenced code', async () => {
    const ctx = makeCtx({ currentCanonicalRel: 'CONFIGURATION.md' });
    const raw = [
      '# T',
      '',
      'See [bench](benchmarks/orukeet-linux-20260918.md).',
      '',
      '```bash',
      '# [not a link](benchmarks/orukeet-linux-20260918.md)',
      'echo "assets/pill-states.png"',
      '```',
      '',
      '![pill](assets/pill-states.png)',
      '',
    ].join('\n');
    const { markdown } = await transformMarkdown('CONFIGURATION.md', raw, ctx);
    const fm = matter(markdown);
    assert.ok(fm.content.includes('/hyprwhspr/benchmarks/orukeet-linux-20260918/'));
    assert.ok(fm.content.includes('/hyprwhspr/docs/assets/pill-states.png'));
    // Code fence preserved verbatim (still contains .md, not rewritten to route).
    assert.ok(fm.content.includes('# [not a link](benchmarks/orukeet-linux-20260918.md)'));
    assert.ok(fm.content.includes('echo "assets/pill-states.png"'));
  });

  it('rewrites reference definitions and preserves code-string examples', async () => {
    const ctx = makeCtx({ currentCanonicalRel: 'CONFIGURATION.md' });
    const raw = [
      '# T',
      '',
      'See [bench][r] and `"code"` with `assets/pill-states.png`.',
      '',
      '[r]: benchmarks/orukeet-linux-20260918.md',
      '',
      '```jsonc',
      '{ "x": "[r]: benchmarks/orukeet-linux-20260918.md" }',
      '```',
      '',
    ].join('\n');
    const { markdown } = await transformMarkdown('CONFIGURATION.md', raw, ctx);
    assert.ok(markdown.includes('/hyprwhspr/benchmarks/orukeet-linux-20260918/'));
    assert.ok(markdown.includes('{ "x": "[r]: benchmarks/orukeet-linux-20260918.md" }'));
  });

  it('rewrites raw HTML href/src where supported, leaves code untouched', async () => {
    const ctx = makeCtx({ currentCanonicalRel: 'CONFIGURATION.md' });
    const raw = [
      '# T',
      '',
      '<a href="benchmarks/orukeet-linux-20260918.md">bench</a>',
      '',
      '<img src="assets/pill-states.png" alt="pill">',
      '',
      '```html',
      '<a href="benchmarks/orukeet-linux-20260918.md">code</a>',
      '```',
      '',
    ].join('\n');
    const { markdown } = await transformMarkdown('CONFIGURATION.md', raw, ctx);
    assert.ok(markdown.includes('<a href="/hyprwhspr/benchmarks/orukeet-linux-20260918/">'));
    assert.ok(markdown.includes('<img src="/hyprwhspr/docs/assets/pill-states.png"'));
    assert.ok(markdown.includes('<a href="benchmarks/orukeet-linux-20260918.md">code</a>'));
  });

  it('preserves canonical fragment links verbatim (no invented anchors)', async () => {
    const ctx = makeCtx({ currentCanonicalRel: 'CONFIGURATION.md' });
    const raw = '# T\n\nSee [a](#setup-1) and [b](#gnomemutter-notes) and [c](#trailing-space).\n';
    const { markdown } = await transformMarkdown('CONFIGURATION.md', raw, ctx);
    assert.ok(markdown.includes('(#setup-1)'));
    assert.ok(markdown.includes('(#gnomemutter-notes)'));
    assert.ok(markdown.includes('(#trailing-space)'));
    assert.equal(ctx.warnings.length, 0);
  });

  it('derives description safely and sets per-page editUrl', async () => {
    const ctx = makeCtx();
    const raw = '# Hello: "world"\n\nFirst paragraph with *emphasis*.\n';
    const { description, markdown } = await transformMarkdown('SUB/FILE.md', raw, ctx);
    assert.ok(description.length > 0);
    const fm = matter(markdown);
    assert.equal(fm.data.editUrl, 'https://github.com/goodroot/hyprwhspr/edit/main/docs/SUB/FILE.md');
    // Round-trips through YAML safely.
    assert.equal(matter(markdown).data.title, 'Hello: "world"');
  });
});

describe('generate() with temp fixtures', () => {
  let tmp = '';
  let canonicalRoot = '';
  let generatedRoot = '';
  let publicRoot = '';
  beforeEach(async () => {
    tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'docs-site-test-'));
    canonicalRoot = path.join(tmp, 'docs');
    generatedRoot = path.join(tmp, 'out', 'docs');
    publicRoot = path.join(tmp, 'out', 'public');
    await fs.mkdir(path.join(canonicalRoot, 'benchmarks'), { recursive: true });
    await fs.mkdir(path.join(canonicalRoot, 'assets'), { recursive: true });
  });
  afterEach(async () => {
    await fs.rm(tmp, { recursive: true, force: true });
  });

  async function writeCanonical() {
    await fs.writeFile(path.join(canonicalRoot, 'CONFIGURATION.md'), '# Config\n\nLink [b](benchmarks/nested.md#frag).\n\n![i](assets/x.png)\n');
    await fs.writeFile(path.join(canonicalRoot, 'benchmarks', 'nested.md'), '# Nested\n\nReceipt [r](nested.json).\n');
    await fs.writeFile(path.join(canonicalRoot, 'benchmarks', 'nested.json'), '{"ok":true}');
    await fs.writeFile(path.join(canonicalRoot, 'assets', 'x.png'), Buffer.from([0x89, 0x50]));
  }

  it('maps nested paths, copies assets, and emits deterministic output', async () => {
    await writeCanonical();
    const first = await generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/hyprwhspr' });
    assert.deepEqual(first.canonicalRels.sort(), ['CONFIGURATION.md', 'benchmarks/nested.md']);
    const genA = await fs.readFile(path.join(generatedRoot, 'configuration.md'), 'utf8');
    assert.ok(genA.includes('/hyprwhspr/benchmarks/nested/#frag'));
    assert.ok(genA.includes('/hyprwhspr/docs/assets/x.png'));
    const genB = await fs.readFile(path.join(generatedRoot, 'benchmarks', 'nested.md'), 'utf8');
    assert.ok(genB.includes('/hyprwhspr/docs/benchmarks/nested.json'));
    // Assets mirrored.
    assert.ok((await fs.stat(path.join(publicRoot, 'assets', 'x.png'))).isFile());
    assert.ok((await fs.stat(path.join(publicRoot, 'benchmarks', 'nested.json'))).isFile());
    const snapshot = new Map();
    for (const rel of ['configuration.md', 'benchmarks/nested.md']) {
      snapshot.set(rel, await fs.readFile(path.join(generatedRoot, ...rel.split('/')), 'utf8'));
    }
    const second = await generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/hyprwhspr' });
    for (const rel of ['configuration.md', 'benchmarks/nested.md']) {
      assert.equal(await fs.readFile(path.join(generatedRoot, ...rel.split('/')), 'utf8'), snapshot.get(rel));
    }
    assert.deepEqual(second.canonicalRels.sort(), first.canonicalRels.sort());
  });

  it('deletes stale generated files and stale assets', async () => {
    await writeCanonical();
    await generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/hyprwhspr' });
    assert.ok((await fs.stat(path.join(generatedRoot, 'configuration.md'))).isFile());
    await fs.unlink(path.join(canonicalRoot, 'benchmarks', 'nested.md'));
    await fs.unlink(path.join(canonicalRoot, 'benchmarks', 'nested.json'));
    const result = await generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/hyprwhspr' });
    let missing = false;
    try {
      await fs.stat(path.join(generatedRoot, 'benchmarks', 'nested.md'));
    } catch {
      missing = true;
    }
    assert.ok(missing, 'stale generated markdown deleted');
    let assetMissing = false;
    try {
      await fs.stat(path.join(publicRoot, 'benchmarks', 'nested.json'));
    } catch {
      assetMissing = true;
    }
    assert.ok(assetMissing, 'stale mirrored asset deleted');
    assert.ok(result.warnings.some((w) => w.includes('stale')));
  });

  it('never touches handwritten index.mdx', async () => {
    await writeCanonical();
    await fs.mkdir(generatedRoot, { recursive: true });
    await fs.writeFile(path.join(generatedRoot, 'index.mdx'), 'handwritten');
    await generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/hyprwhspr' });
    assert.equal(await fs.readFile(path.join(generatedRoot, 'index.mdx'), 'utf8'), 'handwritten');
  });

  it('heading anchor parity: duplicates and punctuation survive generation', async () => {
    await fs.writeFile(
      path.join(canonicalRoot, 'CONFIGURATION.md'),
      '# Guide\n\n## Setup\n\n## Setup\n\n### Available models\n\n### Available models\n\n#### Cohere 🇨🇦\n\nSee [a](#setup) and [b](#setup-1) and [c](#available-models-1).\n',
    );
    await generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/hyprwhspr' });
    const out = await fs.readFile(path.join(generatedRoot, 'configuration.md'), 'utf8');
    const tree = unified().use(remarkParse).parse(matter(out).content);
    const headings = [];
    visit(tree, 'heading', (node) => headings.push({ depth: node.depth, text: mdastToString(node) }));
    assert.ok(headings.some((h) => h.text === 'Setup'));
    assert.equal(headings.filter((h) => h.text === 'Setup').length, 2);
    assert.ok(headings.some((h) => h.text === 'Cohere 🇨🇦'));
    assert.ok(out.includes('(#setup)') && out.includes('(#setup-1)') && out.includes('(#available-models-1)'));
  });

  it('query strings survive .md route rewrites', async () => {
    await fs.writeFile(path.join(canonicalRoot, 'CONFIGURATION.md'), '# T\n');
    await fs.writeFile(path.join(canonicalRoot, 'benchmarks', 'nested.md'), '# N\n');
    const result = await generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/hyprwhspr' });
    assert.ok(result.canonicalRels.includes('benchmarks/nested.md'));
    const ctx = makeCtx({ currentCanonicalRel: 'CONFIGURATION.md' });
    ctx.routeMapLower.set('benchmarks/nested.md', 'benchmarks/nested');
    assert.equal(
      rewriteUrl('benchmarks/nested.md?x=1#frag', ctx),
      '/hyprwhspr/benchmarks/nested/?x=1#frag',
    );
  });

  it('escape outside docs root warns and preserves', async () => {
    const ctx = makeCtx({ currentCanonicalRel: 'benchmarks/nested.md', allowMissingAssets: false });
    const out = rewriteUrl('../../escape.md', ctx);
    assert.equal(out, '../../escape.md');
    assert.ok(ctx.warnings.some((w) => w.startsWith('escape:')));
  });
});

describe('collision detection (before any writes)', () => {
  let tmp = '';
  let canonicalRoot = '';
  let generatedRoot = '';
  let publicRoot = '';
  beforeEach(async () => {
    tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'docs-site-collision-'));
    canonicalRoot = path.join(tmp, 'docs');
    generatedRoot = path.join(tmp, 'out', 'docs');
    publicRoot = path.join(tmp, 'out', 'public');
    await fs.mkdir(canonicalRoot, { recursive: true });
  });
  afterEach(async () => {
    await fs.rm(tmp, { recursive: true, force: true });
  });

  it('rejects underscore-hyphen collision (FOO_BAR vs FOO-BAR)', async () => {
    await fs.writeFile(path.join(canonicalRoot, 'FOO_BAR.md'), '# A\n');
    await fs.writeFile(path.join(canonicalRoot, 'FOO-BAR.md'), '# B\n');
    await assert.rejects(
      generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/' }),
      /route-collision/,
    );
    // No output writes happened.
    let exists = true;
    try {
      await fs.stat(path.join(generatedRoot, 'foo-bar.md'));
    } catch {
      exists = false;
    }
    assert.equal(exists, false);
  });

  it('rejects case-only collision (FOO.md vs foo.md)', async () => {
    await fs.writeFile(path.join(canonicalRoot, 'FOO.md'), '# A\n');
    await fs.writeFile(path.join(canonicalRoot, 'foo.md'), '# B\n');
    await assert.rejects(
      generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/' }),
      /route-collision/,
    );
  });

  it('rejects canonical INDEX.md vs handwritten index.mdx', async () => {
    await fs.writeFile(path.join(canonicalRoot, 'INDEX.md'), '# Home\n');
    await assert.rejects(
      generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/' }),
      /handwritten.*index\.mdx/i,
    );
  });
});

describe('markdown-only policy (.mdx rejected)', () => {
  let tmp = '';
  let canonicalRoot = '';
  let generatedRoot = '';
  let publicRoot = '';
  beforeEach(async () => {
    tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'docs-site-mdx-'));
    canonicalRoot = path.join(tmp, 'docs');
    generatedRoot = path.join(tmp, 'out', 'docs');
    publicRoot = path.join(tmp, 'out', 'public');
    await fs.mkdir(canonicalRoot, { recursive: true });
  });
  afterEach(async () => {
    await fs.rm(tmp, { recursive: true, force: true });
  });

  it('throws unsupported-format and never mirrors .mdx as asset', async () => {
    await fs.writeFile(path.join(canonicalRoot, 'BAR.md'), '# Bar\n');
    await fs.writeFile(path.join(canonicalRoot, 'FOO.mdx'), '# Foo\n');
    await assert.rejects(
      generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/' }),
      /unsupported-format.*\.mdx/i,
    );
    let mirrored = true;
    try {
      await fs.stat(path.join(publicRoot, 'FOO.mdx'));
    } catch {
      mirrored = false;
    }
    assert.equal(mirrored, false);
  });
});

describe('frontmatter adaptation', () => {
  it('preserves safe canonical fields, generated title/description/editUrl win', async () => {
    const ctx = makeCtx();
    const raw = [
      '---',
      'sidebar:',
      '  order: 5',
      'custom: keepme',
      'title: Old Title Should Lose',
      'editUrl: https://evil.example/should-lose',
      '---',
      '# Real Title',
      '',
      'First para body.',
      '',
    ].join('\n');
    const { title, description, markdown } = await transformMarkdown('SUB/FILE.md', raw, ctx);
    assert.equal(title, 'Real Title');
    const fm = matter(markdown);
    assert.equal(fm.data.title, 'Real Title');
    assert.equal(fm.data.editUrl, 'https://github.com/goodroot/hyprwhspr/edit/main/docs/SUB/FILE.md');
    // Preserved safe fields.
    assert.deepEqual(fm.data.sidebar, { order: 5 });
    assert.equal(fm.data.custom, 'keepme');
    // Description derived from body (no canonical description to compete).
    assert.ok(typeof fm.data.description === 'string' && fm.data.description.includes('First para body'));
    assert.equal(description, fm.data.description);
  });

  it('canonical frontmatter description feeds the generated description', async () => {
    const ctx = makeCtx();
    const raw = ['---', 'description: Keep this summary', '---', '# Real Title', '', 'Body.', ''].join('\n');
    const { description, markdown } = await transformMarkdown('X.md', raw, ctx);
    assert.equal(description, 'Keep this summary');
    assert.equal(matter(markdown).data.description, 'Keep this summary');
  });
});

describe('H1 title semantics', () => {
  it('warns on misplaced H1 and keeps strip-only-leading-H1', async () => {
    const ctx = makeCtx();
    const raw = 'Intro para.\n\n# Late Title\n\nBody.\n';
    const { title, markdown } = await transformMarkdown('X.md', raw, ctx);
    // Filename fallback (no leading H1, no frontmatter title).
    assert.equal(title, 'X');
    const fm = matter(markdown);
    // Misplaced H1 retained (never strip arbitrary sections).
    assert.ok(fm.content.includes('# Late Title'));
    assert.ok(ctx.warnings.some((w) => w.startsWith('misplaced-h1:')));
  });
});

describe('asset case handling', () => {
  it('resolves to exact on-disk casing with warning, never wrong-case URL', () => {
    const ctx = makeCtx({
      currentCanonicalRel: 'CONFIGURATION.md',
      existingAssetsExact: new Set(['assets/Pill-States.PNG']),
      existingAssetsLowerMap: new Map([['assets/pill-states.png', 'assets/Pill-States.PNG']]),
      existingAssetsLower: new Set(['assets/pill-states.png']),
      allowMissingAssets: false,
    });
    const out = rewriteUrl('assets/pill-states.png', ctx);
    assert.equal(out, '/hyprwhspr/docs/assets/Pill-States.PNG');
    assert.ok(ctx.warnings.some((w) => w.startsWith('case-mismatch-asset:')));
  });

  it('exact-case links emit no case warning', () => {
    const ctx = makeCtx({
      currentCanonicalRel: 'CONFIGURATION.md',
      existingAssetsExact: new Set(['assets/Pill-States.PNG']),
      existingAssetsLowerMap: new Map([['assets/pill-states.png', 'assets/Pill-States.PNG']]),
      existingAssetsLower: new Set(['assets/pill-states.png']),
      allowMissingAssets: false,
    });
    const out = rewriteUrl('assets/Pill-States.PNG', ctx);
    assert.equal(out, '/hyprwhspr/docs/assets/Pill-States.PNG');
    assert.ok(!ctx.warnings.some((w) => w.startsWith('case-mismatch-asset:')));
  });
});

describe('raw HTML srcset + comments', () => {
  it('rewrites every srcset candidate, leaves comments untouched', async () => {
    const ctx = makeCtx({
      currentCanonicalRel: 'CONFIGURATION.md',
      existingAssetsExact: new Set(['assets/a.png', 'assets/b.png']),
      existingAssetsLowerMap: new Map([
        ['assets/a.png', 'assets/a.png'],
        ['assets/b.png', 'assets/b.png'],
      ]),
      existingAssetsLower: new Set(['assets/a.png', 'assets/b.png']),
      allowMissingAssets: true,
    });
    const raw = [
      '# T',
      '',
      '<img srcset="assets/a.png 1x, assets/b.png 2x" src="assets/a.png" alt="x">',
      '',
      '<!-- <img src="assets/a.png" srcset="assets/a.png 1x"> -->',
      '',
    ].join('\n');
    const { markdown } = await transformMarkdown('CONFIGURATION.md', raw, ctx);
    assert.ok(markdown.includes('srcset="/hyprwhspr/docs/assets/a.png 1x, /hyprwhspr/docs/assets/b.png 2x"'));
    assert.ok(markdown.includes('src="/hyprwhspr/docs/assets/a.png"'));
    // Comment preserved verbatim (no rewrite inside comments).
    assert.ok(markdown.includes('<!-- <img src="assets/a.png" srcset="assets/a.png 1x"> -->'));
  });
});

describe('GFM tables and tasks', () => {
  it('parses GFM tables/tasks: links rewritten, structure preserved', async () => {
    const ctx = makeCtx({
      currentCanonicalRel: 'CONFIGURATION.md',
      existingAssetsExact: new Set(['assets/pill-states.png']),
      existingAssetsLowerMap: new Map([['assets/pill-states.png', 'assets/pill-states.png']]),
      allowMissingAssets: true,
    });
    const raw = [
      '# T',
      '',
      '| Name | Link |',
      '| --- | --- |',
      '| bench | [bench](benchmarks/orukeet-linux-20260918.md) |',
      '| pill | ![pill](assets/pill-states.png) |',
      '',
      '- [ ] task with [bench](benchmarks/orukeet-linux-20260918.md)',
      '- [x] done task',
      '',
    ].join('\n');
    const { markdown } = await transformMarkdown('CONFIGURATION.md', raw, ctx);
    assert.ok(markdown.includes('/hyprwhspr/benchmarks/orukeet-linux-20260918/'));
    assert.ok(markdown.includes('/hyprwhspr/docs/assets/pill-states.png'));
    // Table + task syntax preserved (GFM round-trip).
    assert.ok(markdown.includes('|'));
    assert.ok(markdown.includes('- [ ]') && markdown.includes('- [x]'));
  });
});

describe('deterministic codepoint ordering', () => {
  it('uses codepoint comparison, not locale', () => {
    assert.equal(compareCodepoints('Z', 'a'), -1);
    assert.equal(compareCodepoints('a', 'Z'), 1);
    assert.equal(compareCodepoints('a', 'a'), 0);
    const sorted = ['a', 'Z'].sort(compareCodepoints);
    assert.deepEqual(sorted, ['Z', 'a']);
  });
});

describe('effective Starlight routes (installed loader parity)', () => {
  it('nested INDEX.md strips to its directory; root index -> ""', async () => {
    // Matches the installed routing code sequentially (not a strip-all loop):
    // astro/dist/content/utils.js getContentEntryIdAndSlug
    // (github-slugger per segment + ONE trailing /index strip) plus
    // Starlight normalizeIndexSlug (exact `index` -> ``) plus
    // Starlight slugToParam (`index`/`''`/`/` -> root, else ONE trailing
    // /index strip + `.normalize()`).
    const { slug: githubSlug } = await import('github-slugger');
    function installedEffective(generatedRel) {
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
    assert.equal(generatedToEffectiveRoute('guide/index.md'), 'guide');
    assert.equal(generatedToEffectiveRoute('GUIDE/INDEX.md'.toLowerCase().replace(/_/g, '-')), 'guide');
    assert.equal(generatedToEffectiveRoute('index.md'), '');
    assert.equal(generatedToEffectiveRoute('benchmarks/index.md'), 'benchmarks');
    assert.equal(generatedToEffectiveRoute('configuration.md'), 'configuration');
    // Repeated trailing INDEX: exactly one strip per stage, not strip-all.
    // GUIDE/INDEX/INDEX.md -> Astro `guide/index` -> Starlight `guide`.
    assert.equal(generatedToEffectiveRoute('guide/index/index.md'), 'guide');
    // GUIDE/INDEX/INDEX/INDEX.md -> Astro `guide/index/index` -> Starlight `guide/index`.
    assert.equal(generatedToEffectiveRoute('guide/index/index/index.md'), 'guide/index');
    // Root parity: INDEX/INDEX.md also claims `/` like INDEX.md.
    assert.equal(generatedToEffectiveRoute('index/index.md'), '');
    // Unicode: composed vs decomposed converge only after `.normalize()`.
    assert.equal(generatedToEffectiveRoute('café.md'), 'café');
    assert.equal(generatedToEffectiveRoute('café.md'), 'café');
    assert.equal(generatedToEffectiveRoute('café.md'), generatedToEffectiveRoute('café.md'));
    // Generator matches the installed loader exactly for these shapes.
    for (const g of [
      'guide/index.md',
      'guide/index/index.md',
      'guide/index/index/index.md',
      'index.md',
      'index/index.md',
      'benchmarks/index.md',
      'configuration.md',
      'café.md',
      'café.md',
    ]) {
      assert.equal(generatedToEffectiveRoute(g), installedEffective(g));
    }
    assert.equal(canonicalToEffectiveRoute('GUIDE/INDEX.md'), 'guide');
    assert.equal(canonicalToEffectiveRoute('GUIDE.md'), 'guide');
    assert.equal(canonicalToRoute('GUIDE/INDEX.md'), 'guide');
    assert.equal(canonicalToEffectiveRoute('GUIDE/INDEX/INDEX.md'), 'guide');
    assert.equal(canonicalToEffectiveRoute('GUIDE/INDEX/INDEX/INDEX.md'), 'guide/index');
  });
});

describe('nested INDEX link routing (temp fixture)', () => {
  let tmp = '';
  let canonicalRoot = '';
  let generatedRoot = '';
  let publicRoot = '';
  beforeEach(async () => {
    tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'docs-site-nested-index-'));
    canonicalRoot = path.join(tmp, 'docs');
    generatedRoot = path.join(tmp, 'out', 'docs');
    publicRoot = path.join(tmp, 'out', 'public');
    await fs.mkdir(path.join(canonicalRoot, 'GUIDE'), { recursive: true });
  });
  afterEach(async () => {
    await fs.rm(tmp, { recursive: true, force: true });
  });

  it('links to GUIDE/INDEX.md use effective /guide/ (file stays nested)', async () => {
    await fs.writeFile(path.join(canonicalRoot, 'GUIDE', 'INDEX.md'), '# Nested Guide\n\nBody.\n');
    await fs.writeFile(
      path.join(canonicalRoot, 'CONFIGURATION.md'),
      '# Config\n\nSee [nested](GUIDE/INDEX.md) and [frag](GUIDE/INDEX.md#section).\n',
    );
    await generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/hyprwhspr' });
    // Output file keeps nested index.md path (one page per source file).
    const nestedOut = await fs.readFile(path.join(generatedRoot, 'guide', 'index.md'), 'utf8');
    assert.ok(matter(nestedOut).data.title.includes('Nested Guide'));
    // Internal links use the effective Starlight route, not guide/index.
    const configOut = await fs.readFile(path.join(generatedRoot, 'configuration.md'), 'utf8');
    assert.ok(configOut.includes('/hyprwhspr/guide/'));
    assert.ok(configOut.includes('/hyprwhspr/guide/#section'));
    assert.ok(!configOut.includes('/hyprwhspr/guide/index/'));
  });
});

describe('effective-route collisions (before any writes)', () => {
  let tmp = '';
  let canonicalRoot = '';
  let generatedRoot = '';
  let publicRoot = '';
  beforeEach(async () => {
    tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'docs-site-effective-collision-'));
    canonicalRoot = path.join(tmp, 'docs');
    generatedRoot = path.join(tmp, 'out', 'docs');
    publicRoot = path.join(tmp, 'out', 'public');
    await fs.mkdir(canonicalRoot, { recursive: true });
  });
  afterEach(async () => {
    await fs.rm(tmp, { recursive: true, force: true });
  });

  it('rejects GUIDE.md vs GUIDE/INDEX.md (same effective guide)', async () => {
    await fs.writeFile(path.join(canonicalRoot, 'GUIDE.md'), '# Guide\n');
    await fs.mkdir(path.join(canonicalRoot, 'GUIDE'), { recursive: true });
    await fs.writeFile(path.join(canonicalRoot, 'GUIDE', 'INDEX.md'), '# Nested\n');
    await assert.rejects(
      generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/' }),
      /route-collision.*effective Starlight route "guide"/,
    );
    // No partial writes.
    for (const rel of ['guide.md', 'guide/index.md']) {
      let exists = true;
      try {
        await fs.stat(path.join(generatedRoot, ...rel.split('/')));
      } catch {
        exists = false;
      }
      assert.equal(exists, false, `no partial write: ${rel}`);
    }
  });

  it('rejects lowercase index.md vs handwritten (parity with INDEX.md)', async () => {
    await fs.writeFile(path.join(canonicalRoot, 'index.md'), '# Home\n');
    await assert.rejects(
      generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/' }),
      /route-collision/,
    );
  });

  it('rejects nested INDEX/INDEX.md vs handwritten root (same effective "/")', async () => {
    await fs.mkdir(path.join(canonicalRoot, 'INDEX'), { recursive: true });
    await fs.writeFile(path.join(canonicalRoot, 'INDEX', 'INDEX.md'), '# Deep Home\n');
    await assert.rejects(
      generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/' }),
      /route-collision.*effective Starlight route "\/" \(root\)/,
    );
    let exists = true;
    try {
      await fs.stat(path.join(generatedRoot, 'index', 'index.md'));
    } catch {
      exists = false;
    }
    assert.equal(exists, false, 'no partial write on nested root collision');
  });
});

describe('repeated trailing INDEX (temp fixtures)', () => {
  let tmp = '';
  let canonicalRoot = '';
  let generatedRoot = '';
  let publicRoot = '';
  beforeEach(async () => {
    tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'docs-site-repeated-index-'));
    canonicalRoot = path.join(tmp, 'docs');
    generatedRoot = path.join(tmp, 'out', 'docs');
    publicRoot = path.join(tmp, 'out', 'public');
    await fs.mkdir(canonicalRoot, { recursive: true });
  });
  afterEach(async () => {
    await fs.rm(tmp, { recursive: true, force: true });
  });

  it('links to GUIDE/INDEX/INDEX.md use effective /guide/ (not /guide/index/)', async () => {
    await fs.mkdir(path.join(canonicalRoot, 'GUIDE', 'INDEX'), { recursive: true });
    await fs.writeFile(path.join(canonicalRoot, 'GUIDE', 'INDEX', 'INDEX.md'), '# Deep Guide\n\nBody.\n');
    await fs.writeFile(
      path.join(canonicalRoot, 'CONFIGURATION.md'),
      '# Config\n\nSee [deep](GUIDE/INDEX/INDEX.md) and [frag](GUIDE/INDEX/INDEX.md#section).\n',
    );
    await generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/hyprwhspr' });
    // Output file keeps the nested path (one page per source file).
    const deepOut = await fs.readFile(path.join(generatedRoot, 'guide', 'index', 'index.md'), 'utf8');
    assert.ok(matter(deepOut).data.title.includes('Deep Guide'));
    const configOut = await fs.readFile(path.join(generatedRoot, 'configuration.md'), 'utf8');
    assert.ok(configOut.includes('/hyprwhspr/guide/'));
    assert.ok(configOut.includes('/hyprwhspr/guide/#section'));
    assert.ok(!configOut.includes('/hyprwhspr/guide/index/'));
  });

  it('triple INDEX links to /guide/index/ (only one strip per stage)', async () => {
    await fs.mkdir(path.join(canonicalRoot, 'GUIDE', 'INDEX', 'INDEX'), { recursive: true });
    await fs.writeFile(
      path.join(canonicalRoot, 'GUIDE', 'INDEX', 'INDEX', 'INDEX.md'),
      '# Triple Guide\n\nBody.\n',
    );
    await fs.writeFile(
      path.join(canonicalRoot, 'CONFIGURATION.md'),
      '# Config\n\nSee [triple](GUIDE/INDEX/INDEX/INDEX.md).\n',
    );
    await generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/hyprwhspr' });
    const tripleOut = await fs.readFile(
      path.join(generatedRoot, 'guide', 'index', 'index', 'index.md'),
      'utf8',
    );
    assert.ok(matter(tripleOut).data.title.includes('Triple Guide'));
    const configOut = await fs.readFile(path.join(generatedRoot, 'configuration.md'), 'utf8');
    assert.ok(configOut.includes('/hyprwhspr/guide/index/'));
  });

  it('rejects GUIDE.md vs GUIDE/INDEX/INDEX.md (same effective guide, no partial writes)', async () => {
    await fs.writeFile(path.join(canonicalRoot, 'GUIDE.md'), '# Guide\n');
    await fs.mkdir(path.join(canonicalRoot, 'GUIDE', 'INDEX'), { recursive: true });
    await fs.writeFile(path.join(canonicalRoot, 'GUIDE', 'INDEX', 'INDEX.md'), '# Deep\n');
    await assert.rejects(
      generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/' }),
      /route-collision.*effective Starlight route "guide"/,
    );
    for (const rel of ['guide.md', 'guide/index/index.md']) {
      let exists = true;
      try {
        await fs.stat(path.join(generatedRoot, ...rel.split('/')));
      } catch {
        exists = false;
      }
      assert.equal(exists, false, `no partial write: ${rel}`);
    }
  });
});

describe('unicode normalization collisions (temp fixtures)', () => {
  let tmp = '';
  let canonicalRoot = '';
  let generatedRoot = '';
  let publicRoot = '';
  beforeEach(async () => {
    tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'docs-site-unicode-collision-'));
    canonicalRoot = path.join(tmp, 'docs');
    generatedRoot = path.join(tmp, 'out', 'docs');
    publicRoot = path.join(tmp, 'out', 'public');
    await fs.mkdir(canonicalRoot, { recursive: true });
  });
  afterEach(async () => {
    await fs.rm(tmp, { recursive: true, force: true });
  });

  it('rejects composed vs decomposed CAFÉ.md (same effective café, no partial writes)', async () => {
    // U+00E9 (composed) vs U+0065 U+0301 (decomposed): distinct on disk,
    // identical after Starlight slugToParam `.normalize()`.
    const composed = 'CAF\u00c9.md';
    const decomposed = 'CAFE\u0301.md';
    assert.notEqual(composed, decomposed);
    assert.equal(composed.normalize(), decomposed.normalize());
    await fs.writeFile(path.join(canonicalRoot, composed), '# Composed\n');
    await fs.writeFile(path.join(canonicalRoot, decomposed), '# Decomposed\n');
    await assert.rejects(
      generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/' }),
      /route-collision.*effective Starlight route "caf\u00e9"/,
    );
    for (const rel of ['caf\u00e9.md', 'cafe\u0301.md']) {
      let exists = true;
      try {
        await fs.stat(path.join(generatedRoot, ...rel.split('/')));
      } catch {
        exists = false;
      }
      assert.equal(exists, false, `no partial write: ${rel}`);
    }
  });
});

describe('frontmatter slug override rejection (before any writes)', () => {
  let tmp = '';
  let canonicalRoot = '';
  let generatedRoot = '';
  let publicRoot = '';
  beforeEach(async () => {
    tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'docs-site-slug-override-'));
    canonicalRoot = path.join(tmp, 'docs');
    generatedRoot = path.join(tmp, 'out', 'docs');
    publicRoot = path.join(tmp, 'out', 'public');
    await fs.mkdir(canonicalRoot, { recursive: true });
  });
  afterEach(async () => {
    await fs.rm(tmp, { recursive: true, force: true });
  });

  it('rejects canonical slug with actionable error and writes nothing', async () => {
    await fs.writeFile(
      path.join(canonicalRoot, 'GUIDE.md'),
      '---\nslug: renamed\n---\n# Guide\n\nBody.\n',
    );
    await assert.rejects(
      generate({ canonicalRoot, generatedRoot, publicDocsRoot: publicRoot, base: '/' }),
      /route-override.*GUIDE\.md.*slug: renamed.*Remove the `slug` field/,
    );
    let exists = true;
    try {
      await fs.stat(path.join(generatedRoot, 'guide.md'));
    } catch {
      exists = false;
    }
    assert.equal(exists, false, 'no partial write on slug override');
  });

  it('transformMarkdown strips slug defensively (never survives)', async () => {
    const ctx = makeCtx({ currentCanonicalRel: 'GUIDE.md' });
    const raw = '---\nslug: renamed\nsidebar:\n  order: 1\n---\n# Guide\n\nBody.\n';
    const { markdown } = await transformMarkdown('GUIDE.md', raw, ctx);
    const fm = matter(markdown);
    assert.ok(!Object.hasOwn(fm.data, 'slug'), 'slug must not survive transformation');
    assert.deepEqual(fm.data.sidebar, { order: 1 });
    assert.equal(fm.data.title, 'Guide');
  });
});
