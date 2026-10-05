// Rewrite relative docs links to /docs/ routes (see README mapping).
// Uses Astro per-segment github slugs plus Starlight index strips.
// Assets resolve from canonical sourcePath, preserving case.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { slug } from 'github-slugger';

const CONTENT = path.join(import.meta.dirname, 'content', 'docs', 'docs');

export const slugPath = (rel) => `${rel.replace(/\.mdx?$/i, '').split('/').map((s) => slug(s).replaceAll('_', '-')).join('/')}.md`;

export const effectiveRoute = (to) => {
  let r = `docs/${to.replace(/\.mdx?$/i, '')}`.normalize();
  if (r === 'docs/index') return 'docs';
  if (r.endsWith('/index')) r = r.slice(0, -6);
  if (r.endsWith('/index')) r = r.slice(0, -6);
  return r.normalize();
};

function rewrite(node, ctx) {
  if (!ctx.fileURL || /^([a-z][a-z\d+.-]*:|\/|#)/i.test(node.url)) return;
  const page = path.relative(CONTENT, fileURLToPath(ctx.fileURL));
  if (page.startsWith('..') || !/\.mdx?$/i.test(page)) return;
  const cut = node.url.search(/[?#]/);
  const target = cut < 0 ? node.url : node.url.slice(0, cut);
  const rest = cut < 0 ? '' : node.url.slice(cut);
  if (!target) return;
  const src = ctx.data?.astro?.frontmatter?.sourcePath;
  const base = typeof src === 'string' ? path.posix.dirname(src.split(path.sep).join('/')) : path.posix.dirname(page.split(path.sep).join('/'));
  const resolved = path.posix.join(base, decodeURIComponent(target));
  if (resolved.startsWith('..')) return;
  ctx.setProperty(node, 'url', /\.mdx?$/i.test(resolved) ? `/${effectiveRoute(slugPath(resolved))}/${rest}` : `/docs/${encodeURI(resolved)}${rest}`);
}

export const docsLinks = { name: 'docs-links', link: rewrite, image: rewrite };
