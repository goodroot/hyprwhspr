// Sync ../docs to Starlight docs/ prefix (H1 -> title, rest untouched).
// Validates all plans before deleting/writing; handwritten index.mdx kept.
// See website/README.md for mapping and collision policy.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { slugPath, effectiveRoute } from '../src/docs-links.mjs';

export { effectiveRoute };

const SITE = path.resolve(import.meta.dirname, '..');
const SOURCE = path.resolve(SITE, '..', 'docs');
const PAGES = path.join(SITE, 'src', 'content', 'docs', 'docs');
const ASSETS = path.join(SITE, 'public', 'docs');
const EDIT_URL = 'https://github.com/goodroot/hyprwhspr/edit/main/docs/';

export function buildPlan(files, handwritten = []) {
  const pages = [];
  const assets = [];
  for (const { rel, text } of files) {
    const low = rel.toLowerCase();
    if (low.endsWith('.mdx')) throw new Error(`docs/${rel}: .mdx rejected, use .md with leading H1`);
    if (!low.endsWith('.md')) {
      assets.push(rel);
      continue;
    }
    if (text?.startsWith('---')) throw new Error(`docs/${rel}: frontmatter rejected, start with "# Title"`);
    const h1 = text?.match(/^# (.+)\n+/);
    if (!h1?.[1]?.trim()) throw new Error(`docs/${rel}: must start with a "# Title" line`);
    pages.push({ name: rel, to: slugPath(rel), title: h1[1].trim(), body: text.slice(h1[0].length) });
  }
  const seenFile = new Set();
  const seenRoute = new Set();
  const htmlOut = [];
  for (const rel of handwritten) {
    const r = effectiveRoute(rel);
    seenRoute.add(r);
    htmlOut.push(`${r}/index.html`);
  }
  const plans = [];
  for (const p of pages) {
    const normFile = p.to.normalize();
    if (seenFile.has(normFile)) throw new Error(`docs/${p.name}: collides at ${p.to}`);
    seenFile.add(normFile);
    if (p.to.includes('//')) throw new Error(`docs/${p.name}: degenerate path ${p.to}`);
    for (const seg of p.to.slice(0, -3).split('/')) {
      if (!seg || seg === '.' || seg === '..' || !/[\p{L}\p{N}]/u.test(seg)) throw new Error(`docs/${p.name}: collides with landing /docs/`);
    }
    const r = effectiveRoute(p.to);
    if (r === 'docs') throw new Error(`docs/${p.name}: collides with landing /docs/`);
    if (r.includes('//') || r.endsWith('/')) throw new Error(`docs/${p.name}: degenerate route ${r}`);
    if (seenRoute.has(r)) throw new Error(`docs/${p.name}: collides at /${r}/`);
    seenRoute.add(r);
    htmlOut.push(`${r}/index.html`);
    plans.push(p);
  }
  const seenAsset = new Set();
  const assetDists = [];
  for (const rel of assets) {
    const key = rel.toLowerCase().normalize();
    if (seenAsset.has(key)) throw new Error(`docs/${rel}: asset collides (case-only duplicate)`);
    seenAsset.add(key);
    if (rel.includes('//')) throw new Error(`docs/${rel}: degenerate asset path`);
    assetDists.push(`docs/${rel}`.normalize());
  }
  const allOut = [...htmlOut, ...assetDists].map((s) => s.toLowerCase().normalize());
  for (let i = 0; i < allOut.length; i++) {
    for (let j = i + 1; j < allOut.length; j++) {
      if (allOut[i] === allOut[j] || allOut[i].startsWith(`${allOut[j]}/`) || allOut[j].startsWith(`${allOut[i]}/`)) {
        throw new Error(`collision: ${allOut[i]} vs ${allOut[j]}`);
      }
    }
  }
  return { pages: plans, assets };
}

export async function syncDocs(source = SOURCE, pagesDir = PAGES, assetsDir = ASSETS) {
  const rels = (await fs.readdir(source, { recursive: true })).sort();
  const files = [];
  for (const rel of rels) {
    const from = path.join(source, rel);
    if (!(await fs.stat(from)).isFile()) continue;
    const name = rel.split(path.sep).join('/');
    if (name.toLowerCase().endsWith('.md')) files.push({ rel: name, text: await fs.readFile(from, 'utf8') });
    else files.push({ rel: name, text: null });
  }
  let handwritten = [];
  try {
    const existing = await fs.readdir(pagesDir, { recursive: true });
    handwritten = existing.map((r) => r.split(path.sep).join('/')).filter((r) => r.toLowerCase().endsWith('.mdx')).map((r) => r.replace(/\.mdx$/i, '.md'));
  } catch {}
  const { pages, assets } = buildPlan(files, handwritten);
  await fs.rm(assetsDir, { recursive: true, force: true });
  for (const rel of (await fs.readdir(pagesDir, { recursive: true })).sort()) {
    if (!rel.toLowerCase().endsWith('.md')) continue;
    await fs.rm(path.join(pagesDir, rel), { force: true });
  }
  for (const p of pages) {
    await fs.mkdir(path.dirname(path.join(pagesDir, p.to)), { recursive: true });
    await fs.writeFile(path.join(pagesDir, p.to), `---\ntitle: ${JSON.stringify(p.title)}\nsourcePath: ${JSON.stringify(p.name)}\neditUrl: ${EDIT_URL}${p.name}\n---\n\n` + p.body);
  }
  for (const rel of assets) await fs.cp(path.join(source, rel), path.join(assetsDir, rel));
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) await syncDocs();
