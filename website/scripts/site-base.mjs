// Single-site base helpers; defaults are root base and hyprwhspr.com.
// Overrides: DOCS_SITE_URL (alias DOCS_SITE), DOCS_BASE_PATH (alias DOCS_BASE).

function stripTrailingSlash(value) {
  if (value.length > 1) return value.replace(/\/+$/, '');
  return value;
}

export function resolveSite() {
  const explicit = (process.env.DOCS_SITE_URL || process.env.DOCS_SITE || '').trim();
  if (explicit) return stripTrailingSlash(explicit);
  return 'https://hyprwhspr.com';
}

export function normalizeBase(raw) {
  if (raw === undefined || raw === null) return undefined;
  let base = String(raw).trim();
  if (base === '' || base === '/') return '/';
  if (!base.startsWith('/')) base = `/${base}`;
  return stripTrailingSlash(base);
}

export function resolveBase() {
  const explicit = process.env.DOCS_BASE_PATH ?? process.env.DOCS_BASE;
  if (explicit !== undefined && String(explicit).trim() !== '') return normalizeBase(explicit);
  return '/';
}
