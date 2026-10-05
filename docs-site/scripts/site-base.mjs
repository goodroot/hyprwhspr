// Shared site/base resolution for docs-site.
// Defaults target GitHub Project Pages: https://<owner>.github.io + /hyprwhspr.
// Owner is derived from GITHUB_REPOSITORY with a known upstream fallback.
// Explicit overrides for custom hosting / root preview:
//   DOCS_SITE_URL (alias DOCS_SITE), DOCS_BASE_PATH (alias DOCS_BASE).
// Workflow may populate these from actions/configure-pages outputs.

const UPSTREAM_REPOSITORY = 'goodroot/hyprwhspr';

export function resolveRepository() {
  const repo = (process.env.GITHUB_REPOSITORY || '').trim();
  if (repo && repo.includes('/')) return repo;
  return UPSTREAM_REPOSITORY;
}

export function resolveOwner() {
  return resolveRepository().split('/')[0] || 'goodroot';
}

function stripTrailingSlash(value) {
  if (value.length > 1) return value.replace(/\/+$/, '');
  return value;
}

export function resolveSite() {
  const explicit =
    (process.env.DOCS_SITE_URL || process.env.DOCS_SITE || '').trim();
  if (explicit) return stripTrailingSlash(explicit);
  return `https://${resolveOwner()}.github.io`;
}

export function normalizeBase(raw) {
  if (raw === undefined || raw === null) return undefined;
  let base = String(raw).trim();
  if (base === '' || base === '/') return '/';
  if (!base.startsWith('/')) base = `/${base}`;
  base = stripTrailingSlash(base);
  return base;
}

export function resolveBase() {
  const explicit = process.env.DOCS_BASE_PATH ?? process.env.DOCS_BASE;
  if (explicit !== undefined && String(explicit).trim() !== '') {
    return normalizeBase(explicit);
  }
  return '/hyprwhspr';
}
