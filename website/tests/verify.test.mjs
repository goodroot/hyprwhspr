// Verifier tests: temp dist fixtures, no network, no repo mutation.
// Covers base, assets, fragments, pagefind, marketing, docs, install.sh.
// New checks are opt-out in legacy cases via check flags.

import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { verify } from '../scripts/verify-build.mjs';

let tmp = '';
let distDir = '';
let canonicalRoot = '';

async function writeHtml(rel, html) {
  const full = path.join(distDir, ...rel.split('/'));
  await fs.mkdir(path.dirname(full), { recursive: true });
  await fs.writeFile(full, html, 'utf8');
}

beforeEach(async () => {
  tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'website-verify-'));
  distDir = path.join(tmp, 'dist');
  canonicalRoot = path.join(tmp, 'docs');
  await fs.mkdir(distDir, { recursive: true });
  await fs.mkdir(canonicalRoot, { recursive: true });
});

afterEach(async () => {
  await fs.rm(tmp, { recursive: true, force: true });
});

describe('verify base-prefix handling', () => {
  it('errors on absolute href missing the project base', async () => {
    await writeHtml('index.html', '<html><body><a href="/configuration/">x</a></body></html>');
    const { errors } = await verify({ distDir, base: '/hyprwhspr', expectedAssets: [], checkPagefind: false, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: false });
    assert.ok(errors.some((e) => e.includes('missing base prefix')));
  });

  it('errors on absolute img src missing the project base (not silently skipped)', async () => {
    await writeHtml('index.html', '<html><body><img src="/docs/assets/x.png"></body></html>');
    const { errors } = await verify({ distDir, base: '/hyprwhspr', expectedAssets: [], checkPagefind: false, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: false });
    assert.ok(errors.some((e) => e.includes('missing base prefix')));
  });

  it('passes base-prefixed internal links', async () => {
    await writeHtml(
      'configuration/index.html',
      '<html><body><h1 id="a">A</h1><a href="/hyprwhspr/managed-installation/">m</a></body></html>',
    );
    await writeHtml('managed-installation/index.html', '<html><body>ok</body></html>');
    const { errors } = await verify({ distDir, base: '/hyprwhspr', expectedAssets: [], checkPagefind: false, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: false });
    assert.deepEqual(errors, []);
  });
});

describe('verify asset coverage', () => {
  it('derives expected assets from canonical root, not a hardcoded sample', async () => {
    await fs.mkdir(path.join(canonicalRoot, 'assets'), { recursive: true });
    await fs.writeFile(path.join(canonicalRoot, 'assets', 'extra.bin'), 'x');
    await writeHtml('index.html', '<html><body>hi</body></html>');
    const { errors, expectedAssets } = await verify({ distDir, base: '/', canonicalRoot, checkPagefind: false, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: false });
    assert.ok(expectedAssets.includes('docs/assets/extra.bin'));
    assert.ok(errors.some((e) => e.includes('missing dist asset: docs/assets/extra.bin')));
  });

  it('passes when every mirrored asset exists', async () => {
    await fs.mkdir(path.join(canonicalRoot, 'assets'), { recursive: true });
    await fs.writeFile(path.join(canonicalRoot, 'assets', 'x.png'), 'x');
    await fs.mkdir(path.join(distDir, 'docs', 'assets'), { recursive: true });
    await fs.writeFile(path.join(distDir, 'docs', 'assets', 'x.png'), 'x');
    await writeHtml('index.html', '<html><body>hi</body></html>');
    const { errors } = await verify({ distDir, base: '/', canonicalRoot, checkPagefind: false, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: false });
    assert.deepEqual(errors, []);
  });
});

describe('verify fragments and pagefind', () => {
  it('errors on invalid same-page and cross-page fragments', async () => {
    await writeHtml('index.html', '<html><body><a href="#nope">x</a></body></html>');
    const r1 = await verify({ distDir, base: '/', expectedAssets: [], checkPagefind: false, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: false });
    assert.ok(r1.errors.some((e) => e.includes('#nope')));

    await writeHtml('index.html', '<html><body><a href="/other/#nope">x</a></body></html>');
    await writeHtml('other/index.html', '<html><body><h1 id="yes">Y</h1></body></html>');
    const r2 = await verify({ distDir, base: '/', expectedAssets: [], checkPagefind: false, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: false });
    assert.ok(r2.errors.some((e) => e.includes('#nope')));
  });

  it('errors when pagefind output is missing, passes when present', async () => {
    await writeHtml('index.html', '<html><body>hi</body></html>');
    const missing = await verify({ distDir, base: '/', expectedAssets: [], checkPagefind: true, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: false });
    assert.ok(missing.errors.some((e) => e.includes('pagefind')));

    await fs.mkdir(path.join(distDir, 'pagefind'), { recursive: true });
    await fs.writeFile(path.join(distDir, 'pagefind', 'pagefind.js'), '// dummy');
    const present = await verify({ distDir, base: '/', expectedAssets: [], checkPagefind: true, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: false });
    assert.ok(!present.errors.some((e) => e.includes('pagefind')));
  });
});

describe('verify single-site routes and install.sh', () => {
  it('requires marketing routes', async () => {
    await writeHtml('index.html', '<html><body>hi</body></html>');
    const r = await verify({ distDir, base: '/', expectedAssets: [], checkPagefind: false, checkMarketing: true, checkDocsRoutes: false, checkInstallSh: false });
    assert.ok(r.errors.some((e) => e.includes('missing marketing route')));
  });

  it('requires docs routes under /docs/ and forbids unprefixed', async () => {
    await writeHtml('docs/index.html', '<html><body>docs</body></html>');
    const missing = await verify({ distDir, base: '/', expectedAssets: [], canonicalRoot, checkPagefind: false, checkMarketing: false, checkDocsRoutes: true, checkInstallSh: false });
    assert.ok(missing.errors.some((e) => e.includes('missing docs route')));
    await writeHtml('docs/configuration/index.html', '<html><body>c</body></html>');
    await writeHtml('docs/managed-installation/index.html', '<html><body>m</body></html>');
    await fs.mkdir(path.join(canonicalRoot, 'benchmarks'), { recursive: true });
    await fs.writeFile(path.join(canonicalRoot, 'benchmarks', 'r.md'), '# R\n');
    await writeHtml('docs/benchmarks/r/index.html', '<html><body>r</body></html>');
    await writeHtml('configuration/index.html', '<html><body>stale</body></html>');
    const bad = await verify({ distDir, base: '/', expectedAssets: [], canonicalRoot, checkPagefind: false, checkMarketing: false, checkDocsRoutes: true, checkInstallSh: false });
    assert.ok(bad.errors.some((e) => e.includes('unprefixed docs route')));
  });

  it('checks install.sh byte parity without executing', async () => {
    const src = path.join(canonicalRoot, 'install.sh');
    await fs.writeFile(src, '#!/bin/sh\necho hi\n');
    await writeHtml('index.html', '<html><body>hi</body></html>');
    const missing = await verify({ distDir, base: '/', expectedAssets: [], checkPagefind: false, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: true, installSh: src });
    assert.ok(missing.errors.some((e) => e.includes('install.sh')));
    await fs.writeFile(path.join(distDir, 'install.sh'), '#!/bin/sh\necho hi\n');
    const ok = await verify({ distDir, base: '/', expectedAssets: [], checkPagefind: false, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: true, installSh: src });
    assert.ok(!ok.errors.some((e) => e.includes('install.sh')));
    await fs.writeFile(path.join(distDir, 'install.sh'), '#!/bin/sh\necho other\n');
    const diff = await verify({ distDir, base: '/', expectedAssets: [], checkPagefind: false, checkMarketing: false, checkDocsRoutes: false, checkInstallSh: true, installSh: src });
    assert.ok(diff.errors.some((e) => e.includes('byte mismatch')));
  });
});
