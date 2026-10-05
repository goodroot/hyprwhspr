// Offline verification for website/dist with docs at /docs/.
// Checks links, fragments, assets, marketing, install.sh, pagefind.
// Usage: node scripts/verify-build.mjs [distDir]

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeBase, resolveBase } from './site-base.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEBSITE_DIR = path.resolve(HERE, '..');
const REPO_ROOT = path.resolve(WEBSITE_DIR, '..');
export const CANONICAL_ROOT = path.join(REPO_ROOT, 'docs');
export const INSTALL_SH = path.join(REPO_ROOT, 'scripts', 'install.sh');

export function compareCodepoints(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

function stripQueryFragment(url) {
  let u = url;
  const hash = u.indexOf('#');
  const frag = hash === -1 ? '' : u.slice(hash + 1);
  if (hash !== -1) u = u.slice(0, hash);
  const q = u.indexOf('?');
  if (q !== -1) u = u.slice(0, q);
  return { path: u, fragment: frag };
}

function extractAttrs(html, tag, attr) {
  const out = [];
  const re = new RegExp(`<${tag}\\b[^>]*?\\b${attr}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s>]+))`, 'gi');
  let m;
  while ((m = re.exec(html)) !== null) out.push(m[1] ?? m[2] ?? m[3] ?? '');
  return out;
}

function extractIds(html) {
  const ids = new Set();
  const re = /\bid\s*=\s*(?:"([^"]*)"|'([^']*)')/gi;
  let m;
  while ((m = re.exec(html)) !== null) ids.add(m[1] ?? m[2]);
  return ids;
}

function isExternal(url) {
  return /^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|\/\/|data:|mailto:|tel:|blob:)/.test(url);
}

async function listHtmlFiles(dir) {
  const out = [];
  async function walk(d) {
    const entries = await fs.readdir(d, { withFileTypes: true });
    entries.sort((a, b) => compareCodepoints(a.name, b.name));
    for (const e of entries) {
      const full = path.join(d, e.name);
      if (e.isDirectory()) await walk(full);
      else if (e.isFile() && e.name.endsWith('.html')) out.push(full);
    }
  }
  await walk(dir);
  return out.sort(compareCodepoints);
}

export async function listCanonicalAssetRels(canonicalRoot) {
  const out = [];
  async function walk(dir) {
    let entries;
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
        const rel = path.relative(canonicalRoot, full).split(path.sep).join('/');
        if (/\.mdx?$/i.test(rel)) continue;
        out.push(rel);
      }
    }
  }
  await walk(canonicalRoot);
  out.sort(compareCodepoints);
  return out;
}

function urlPathToDistFile(distDir, urlPath) {
  let p = decodeURIComponent(urlPath);
  if (p === '' ) return null;
  if (p.endsWith('/')) return path.join(distDir, ...p.split('/').filter(Boolean), 'index.html');
  if (path.extname(p) !== '') return path.join(distDir, ...p.split('/').filter(Boolean));
  return path.join(distDir, ...p.split('/').filter(Boolean), 'index.html');
}

export async function verify(options = {}) {
  const distDir = options.distDir ?? path.join(WEBSITE_DIR, 'dist');
  const base = normalizeBase(options.base ?? resolveBase()) ?? '/';
  const canonicalRoot = options.canonicalRoot ?? CANONICAL_ROOT;
  const installSh = options.installSh ?? INSTALL_SH;
  const checkPagefind = options.checkPagefind ?? true;
  const checkMarketing = options.checkMarketing ?? true;
  const checkDocsRoutes = options.checkDocsRoutes ?? true;
  const checkInstallSh = options.checkInstallSh ?? true;
  const errors = [];
  const warnings = [];
  const htmlFiles = await listHtmlFiles(distDir);
  if (htmlFiles.length === 0) errors.push(`no html files in ${distDir}`);
  const idsByFile = new Map();
  for (const f of htmlFiles) {
    const html = await fs.readFile(f, 'utf8');
    idsByFile.set(f, extractIds(html));
  }
  const basePrefix = base === '/' ? '/' : `${base}/`;
  for (const f of htmlFiles) {
    const html = await fs.readFile(f, 'utf8');
    const relFromDist = path.relative(distDir, f).split(path.sep).join('/');
    const hrefs = extractAttrs(html, 'a', 'href');
    const srcs = [
      ...extractAttrs(html, 'img', 'src'),
      ...extractAttrs(html, 'script', 'src'),
      ...extractAttrs(html, 'link', 'href'),
    ];
    for (const raw of [...hrefs.map((u) => ({ u, kind: 'href' })), ...srcs.map((u) => ({ u, kind: 'src' }))]) {
      const { u, kind } = raw;
      if (!u || u.startsWith('#')) {
        if (u.startsWith('#')) {
          const frag = u.slice(1);
          if (frag !== '' && !idsByFile.get(f)?.has(frag) && !idsByFile.get(f)?.has(decodeURIComponent(frag))) {
            errors.push(`${relFromDist}: ${kind} fragment #${frag} not found in same page`);
          }
        }
        continue;
      }
      if (isExternal(u)) continue;
      const { path: urlPath, fragment } = stripQueryFragment(u);
      if (urlPath === '') {
        if (fragment && !idsByFile.get(f)?.has(fragment)) {
          errors.push(`${relFromDist}: ${kind} fragment #${fragment} not found`);
        }
        continue;
      }
      let distPath = null;
      let targetFile = null;
      if (urlPath.startsWith('/')) {
        if (base !== '/' && !urlPath.startsWith(basePrefix) && urlPath !== base) {
          // Absolute URL missing the base prefix is a hard error.
          errors.push(`${relFromDist}: ${kind} ${u} missing base prefix ${base}`);
          continue;
        }
        const stripped = base === '/' ? urlPath : urlPath.slice(base.length) || '/';
        distPath = urlPathToDistFile(distDir, stripped);
        targetFile = distPath;
      } else {
        const curUrlDir = `/${path.relative(distDir, path.dirname(f)).split(path.sep).join('/')}/`;
        const merged = path.posix.normalize(path.posix.join(curUrlDir, urlPath));
        targetFile = urlPathToDistFile(distDir, merged);
      }
      if (targetFile) {
        try {
          const st = await fs.stat(targetFile);
          if (!st.isFile()) errors.push(`${relFromDist}: ${kind} ${u} -> not a file`);
        } catch {
          errors.push(`${relFromDist}: ${kind} ${u} -> missing file ${path.relative(distDir, targetFile)}`);
          continue;
        }
        if (fragment && targetFile.endsWith('.html')) {
          const ids = idsByFile.get(targetFile);
          if (ids && !ids.has(fragment) && !ids.has(decodeURIComponent(fragment))) {
            errors.push(`${relFromDist}: ${kind} ${u} fragment #${fragment} not found in ${path.relative(distDir, targetFile)}`);
          }
        }
      }
    }
  }
  let expectedAssets;
  if (options.expectedAssets !== undefined) {
    expectedAssets = options.expectedAssets;
  } else {
    const canonicalRels = await listCanonicalAssetRels(canonicalRoot);
    expectedAssets = canonicalRels.map((rel) => `docs/${rel}`);
  }
  for (const rel of expectedAssets) {
    try {
      await fs.stat(path.join(distDir, ...rel.split('/')));
    } catch {
      errors.push(`missing dist asset: ${rel}`);
    }
  }
  // Require marketing routes to stay built.
  if (checkMarketing) {
    for (const rel of ['index.html', 'privacy/index.html', 'linux-speech-to-text/index.html', 'best-speech-to-text-models/index.html', 'dictation-future-programming/index.html']) {
      try {
        await fs.stat(path.join(distDir, ...rel.split('/')));
      } catch {
        errors.push(`missing marketing route: ${rel}`);
      }
    }
  }
  // Require docs routes under /docs/ and forbid unprefixed duplicates.
  if (checkDocsRoutes) {
    for (const rel of ['docs/index.html', 'docs/configuration/index.html', 'docs/managed-installation/index.html']) {
      try {
        await fs.stat(path.join(distDir, ...rel.split('/')));
      } catch {
        errors.push(`missing docs route: ${rel}`);
      }
    }
    try {
      const entries = await fs.readdir(path.join(canonicalRoot, 'benchmarks'));
      let found = false;
      for (const e of entries) {
        if (e.toLowerCase().endsWith('.md')) {
          const stem = e.replace(/\.md$/i, '').toLowerCase().replace(/_/g, '-');
          try {
            await fs.stat(path.join(distDir, 'docs', 'benchmarks', stem, 'index.html'));
            found = true;
          } catch {}
        }
      }
      if (!found) errors.push('missing docs route: docs/benchmarks/<report>/index.html');
    } catch {
      errors.push('missing docs route: docs/benchmarks/<report>/index.html');
    }
    for (const rel of ['configuration/index.html', 'managed-installation/index.html']) {
      try {
        await fs.stat(path.join(distDir, ...rel.split('/')));
        errors.push(`unprefixed docs route must not exist: ${rel}`);
      } catch {}
    }
  }
  // Require copied install.sh to match the canonical script byte-for-byte.
  if (checkInstallSh) {
    try {
      const [a, b] = await Promise.all([fs.readFile(path.join(distDir, 'install.sh')), fs.readFile(installSh)]);
      if (!a.equals(b)) errors.push('install.sh byte mismatch: dist/install.sh differs from scripts/install.sh');
    } catch {
      errors.push('missing dist asset: install.sh (expected copy of scripts/install.sh)');
    }
  }
  // Require Pagefind output.
  const pagefindCandidates = ['pagefind/pagefind.js', 'pagefind/pagefind-ui.js', '_pagefind/pagefind.js'];
  let pagefindFound = false;
  for (const c of pagefindCandidates) {
    try {
      await fs.stat(path.join(distDir, ...c.split('/')));
      pagefindFound = true;
      break;
    } catch {}
  }
  // Accept any non-empty pagefind directory.
  if (!pagefindFound) {
    for (const d of ['pagefind', '_pagefind']) {
      try {
        const entries = await fs.readdir(path.join(distDir, d));
        if (entries.length > 0) {
          pagefindFound = true;
          break;
        }
      } catch {}
    }
  }
  if (!pagefindFound && checkPagefind) errors.push('pagefind output not found (expected pagefind/ or _pagefind/)');
  return { errors, warnings, htmlFiles: htmlFiles.length, base, expectedAssets };
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const distDir = process.argv[2] ?? path.join(WEBSITE_DIR, 'dist');
  verify({ distDir }).then(({ errors, warnings, htmlFiles, base }) => {
    console.log(`verify: ${htmlFiles} html files, base ${base}`);
    for (const w of warnings) console.warn(`verify warning: ${w}`);
    if (errors.length > 0) {
      for (const e of errors) console.error(`verify error: ${e}`);
      process.exit(1);
    }
    console.log('verify: ok');
  });
}
