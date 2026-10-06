// Syncs ../docs into Starlight at /docs/: pages to src/content/docs/docs
// (H1 -> title), everything else to public/docs. Output is gitignored; edit
// docs/ instead.
import fs from 'node:fs/promises';
import path from 'node:path';
import { slugPath } from '../src/docs-links.mjs';

const SITE = path.resolve(import.meta.dirname, '..');
const SOURCE = path.resolve(SITE, '..', 'docs');
const PAGES = path.join(SITE, 'src', 'content', 'docs', 'docs');
const ASSETS = path.join(SITE, 'public', 'docs');
const EDIT_URL = 'https://github.com/goodroot/hyprwhspr/edit/main/docs/';

for (const entry of await fs.readdir(PAGES)) {
  if (entry !== 'index.mdx') await fs.rm(path.join(PAGES, entry), { recursive: true });
}
await fs.rm(ASSETS, { recursive: true, force: true });

const written = new Set();
for (const rel of (await fs.readdir(SOURCE, { recursive: true })).sort()) {
  const from = path.join(SOURCE, rel);
  if (!(await fs.stat(from)).isFile()) continue;
  if (!rel.endsWith('.md')) {
    await fs.cp(from, path.join(ASSETS, rel));
    continue;
  }
  const name = rel.split(path.sep).join('/');
  const to = slugPath(name);
  if (to === 'index.md' || written.has(to)) throw new Error(`docs/${name}: collides with another page at ${to}`);
  written.add(to);

  const text = await fs.readFile(from, 'utf8');
  const h1 = text.match(/^# (.+)\n+/);
  if (!h1) throw new Error(`docs/${name}: must start with a "# Title" line`);
  const front = `---\ntitle: ${JSON.stringify(h1[1].trim())}\neditUrl: ${EDIT_URL}${name}\n---\n\n`;
  await fs.mkdir(path.dirname(path.join(PAGES, to)), { recursive: true });
  await fs.writeFile(path.join(PAGES, to), front + text.slice(h1[0].length));
}
