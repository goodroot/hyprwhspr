// Docs pipeline checks: mapping, collisions, H1/editUrl, link rewrites.
// Temp fixtures only; no network or repo mutation.
import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { slugPath, docsLinks } from '../src/docs-links.mjs';
import { effectiveRoute, buildPlan, syncDocs } from '../scripts/sync-docs.mjs';

const TESTS = path.dirname(new URL(import.meta.url).pathname);
const CONTENT = path.resolve(TESTS, '..', 'src', 'content', 'docs', 'docs');

function fileURL(page) {
  return pathToFileURL(path.join(CONTENT, page)).href;
}

function rewrite(page, url, sourcePath) {
  const node = { url };
  const ctx = { fileURL: fileURL(page), data: sourcePath ? { astro: { frontmatter: { sourcePath } } } : {}, setProperty: (n, k, v) => { n[k] = v; } };
  docsLinks.link(node, ctx);
  return node.url;
}

function md(rel, text) {
  return { rel, text };
}

describe('mapping', () => {
  it('lowercases and kebabs generated paths', () => {
    assert.equal(slugPath('CONFIGURATION.md'), 'configuration.md');
    assert.equal(slugPath('MANAGED_INSTALLATION.md'), 'managed-installation.md');
    assert.equal(slugPath('benchmarks/ORUKEET_Test.md'), 'benchmarks/orukeet-test.md');
  });

  it('computes docs routes with two index strips and NFC', () => {
    assert.equal(effectiveRoute('configuration.md'), 'docs/configuration');
    assert.equal(effectiveRoute('guide/index.md'), 'docs/guide');
    assert.equal(effectiveRoute('guide/index/index.md'), 'docs/guide');
    assert.equal(effectiveRoute('index.md'), 'docs');
    assert.equal('docs/caf\u00e9'.normalize(), 'docs/cafe\u0301'.normalize());
  });
});

describe('buildPlan', () => {
  it('accepts sensible inputs with H1 title and untouched fences', () => {
    const body = 'Intro.\n\n```bash\n# not a title\n```\n';
    const { pages } = buildPlan([md('CONFIGURATION.md', '# Configuration guide\n\n' + body)]);
    assert.equal(pages[0].to, 'configuration.md');
    assert.equal(pages[0].title, 'Configuration guide');
    assert.ok(pages[0].body.includes('# not a title'));
  });

  it('rejects frontmatter, mdx, and missing H1', () => {
    assert.throws(() => buildPlan([md('A.md', '---\ntitle: x\n---\n# Hi\n\nx')]), /frontmatter/);
    assert.throws(() => buildPlan([{ rel: 'A.mdx', text: '# Hi\n\nx' }]), /\.mdx/);
    assert.throws(() => buildPlan([md('A.md', 'No title\n\nbody')]), /"# Title"/);
  });

  it('rejects landing collisions including INDEX and unsluggable', () => {
    assert.throws(() => buildPlan([md('INDEX.md', '# T\n\nx')]), /landing/);
    assert.throws(() => buildPlan([md('INDEX/INDEX.md', '# T\n\nx')]), /landing/);
    assert.throws(() => buildPlan([md('!.md', '# T\n\nx')]), /landing/);
  });

  it('rejects duplicate effective routes and NFC variants', () => {
    assert.throws(() => buildPlan([md('GUIDE.md', '# A\n\nx'), md('GUIDE/INDEX.md', '# B\n\nx')]), /collides/);
    assert.throws(() => buildPlan([md('caf\u00e9.md', '# A\n\nx'), md('cafe\u0301.md', '# B\n\nx')]), /collides/);
  });

  it('rejects asset shadows, ancestors, and case duplicates', () => {
    assert.throws(() => buildPlan([md('GUIDE.md', '# A\n\nx'), { rel: 'guide/index.html', text: null }]), /collision/);
    assert.throws(() => buildPlan([md('GUIDE.md', '# A\n\nx'), { rel: 'guide', text: null }]), /collision/);
    assert.throws(() => buildPlan([{ rel: 'a.png', text: null }, { rel: 'A.PNG', text: null }]), /case-only/);
  });

  it('rejects Astro-equivalent punctuation and space variants', () => {
    assert.throws(() => buildPlan([md('A.md', '# A\n\nx'), md('A!.md', '# B\n\nx')]), /collides/);
    assert.throws(() => buildPlan([md('A B.md', '# A\n\nx'), md('A-B.md', '# B\n\nx')]), /collides/);
  });

  it('protects handwritten nested indexes', () => {
    assert.throws(() => buildPlan([md('GUIDE.md', '# A\n\nx')], ['guide/index.md']), /collides/);
  });
});

describe('docsLinks', () => {
  it('prefixes md links and assets under /docs/', () => {
    assert.equal(rewrite('configuration.md', 'MANAGED_INSTALLATION.md'), '/docs/managed-installation/');
    assert.equal(rewrite('configuration.md', 'benchmarks/orukeet-linux-20260918.md'), '/docs/benchmarks/orukeet-linux-20260918/');
    assert.equal(rewrite('configuration.md', 'assets/pill-states.png'), '/docs/assets/pill-states.png');
    assert.equal(rewrite('benchmarks/orukeet-linux-20260918.md', 'orukeet-linux-20260918.json'), '/docs/benchmarks/orukeet-linux-20260918.json');
  });

  it('keeps query/hash and skips external, absolute, hash, escape', () => {
    assert.equal(rewrite('configuration.md', 'MANAGED_INSTALLATION.md#sec?x=1'), '/docs/managed-installation/#sec?x=1');
    assert.equal(rewrite('configuration.md', 'https://example.com/x.md'), 'https://example.com/x.md');
    assert.equal(rewrite('configuration.md', '/docs/managed-installation/'), '/docs/managed-installation/');
    assert.equal(rewrite('configuration.md', '#minimal-configuration'), '#minimal-configuration');
    assert.equal(rewrite('configuration.md', '../outside.md'), '../outside.md');
  });

  it('strips nested index routes and normalizes NFC destinations', () => {
    assert.equal(rewrite('configuration.md', 'GUIDE/INDEX.md#x'), '/docs/guide/#x');
    assert.equal(rewrite('configuration.md', 'guide/index/index.md'), '/docs/guide/');
    assert.equal(rewrite('configuration.md', 'caf\u00e9.md'), rewrite('configuration.md', 'cafe\u0301.md'));
  });

  it('resolves assets from canonical source keep case underscore', () => {
    assert.equal(rewrite('foo-bar/page.md', 'Picture.png', 'foo_bar/PAGE.md'), '/docs/foo_bar/Picture.png');
    assert.equal(rewrite('foo-bar/page.md', 'OTHER.md', 'foo_bar/PAGE.md'), '/docs/foo-bar/other/');
  });
});

describe('syncDocs', () => {
  let tmp = '';
  let source = '';
  let pages = '';
  let assets = '';
  beforeEach(async () => {
    tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'website-docs-'));
    source = path.join(tmp, 'docs');
    pages = path.join(tmp, 'pages');
    assets = path.join(tmp, 'public');
    await fs.mkdir(source, { recursive: true });
    await fs.mkdir(pages, { recursive: true });
    await fs.writeFile(path.join(pages, 'index.mdx'), '---\ntitle: t\n---\nlanding');
  });
  afterEach(async () => {
    await fs.rm(tmp, { recursive: true, force: true });
  });

  it('writes canonical title, editUrl, and untouched fences', async () => {
    await fs.writeFile(path.join(source, 'CONFIGURATION.md'), '# Configuration guide\n\n```bash\n# keep\n```\n');
    await fs.mkdir(path.join(source, 'assets'), { recursive: true });
    await fs.writeFile(path.join(source, 'assets', 'p.png'), 'img');
    await syncDocs(source, pages, assets);
    const out = await fs.readFile(path.join(pages, 'configuration.md'), 'utf8');
    assert.ok(out.includes('title: "Configuration guide"'));
    assert.ok(out.includes('editUrl: https://github.com/goodroot/hyprwhspr/edit/main/docs/CONFIGURATION.md'));
    assert.ok(out.includes('# keep'));
    assert.ok((await fs.stat(path.join(assets, 'assets', 'p.png'))).isFile());
  });

  it('validates before deleting and keeps handwritten mdx', async () => {
    await fs.writeFile(path.join(pages, 'keep.md'), 'old');
    await fs.mkdir(path.join(pages, 'guide'), { recursive: true });
    await fs.writeFile(path.join(pages, 'guide', 'index.mdx'), 'hand');
    await fs.writeFile(path.join(source, 'GUIDE.md'), '# A\n\nx');
    await fs.mkdir(path.join(source, 'GUIDE'), { recursive: true });
    await fs.writeFile(path.join(source, 'GUIDE', 'INDEX.md'), '# B\n\nx');
    await assert.rejects(() => syncDocs(source, pages, assets), /collides/);
    assert.equal(await fs.readFile(path.join(pages, 'keep.md'), 'utf8'), 'old');
    assert.equal(await fs.readFile(path.join(pages, 'guide', 'index.mdx'), 'utf8'), 'hand');
  });
});
