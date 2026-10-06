// Rewrites relative links in pages synced from ../docs (scripts/sync-docs.mjs):
// FOO_BAR.md#x -> /docs/foo-bar/#x; anything else -> /docs/<path>, mirrored
// into public/docs. Assets resolve through the lowercased page directory, so
// keep docs/ subdirectory names lowercase.
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const CONTENT = path.join(import.meta.dirname, 'content', 'docs', 'docs');

export const slugPath = (rel) => rel.toLowerCase().replaceAll('_', '-');

function rewrite(node, ctx) {
  if (!ctx.fileURL || /^([a-z][a-z\d+.-]*:|\/|#)/i.test(node.url)) return;
  const page = path.relative(CONTENT, fileURLToPath(ctx.fileURL));
  if (page.startsWith('..') || !page.endsWith('.md')) return;

  const cut = node.url.search(/[?#]/);
  const target = cut < 0 ? node.url : node.url.slice(0, cut);
  const rest = cut < 0 ? '' : node.url.slice(cut);
  const dir = path.posix.dirname(page.split(path.sep).join('/'));
  const resolved = path.posix.join(dir, decodeURIComponent(target));
  if (resolved.startsWith('..')) return;
  ctx.setProperty(node, 'url', resolved.endsWith('.md')
    ? `/docs/${slugPath(resolved).slice(0, -3)}/${rest}`
    : `/docs/${encodeURI(resolved)}${rest}`);
}

export const docsLinks = { name: 'docs-links', link: rewrite, image: rewrite };
