"""Exact historical generated-unit recognition; no Git or systemd dependency."""
import json
from pathlib import Path


def generated_unit(path, root):
    path, root = Path(path), Path(root)
    if path.is_symlink() or not path.is_file():
        return False
    try:
        content = path.read_text(encoding='utf-8')
        template = root / 'config/systemd' / path.name
        if template.is_file() and content == template.read_text(encoding='utf-8').replace('/usr/lib/hyprwhspr', str(root)):
            return True
        catalog_path = Path(__file__).resolve().parents[2] / 'share/legacy-systemd-units.json'
        catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
        if not isinstance(catalog, dict) or catalog.get('format') != 1:
            return False
        return any(content == entry['content'].replace('/usr/lib/hyprwhspr', str(root))
                   for entry in catalog['units'].get(path.name, []))
    except (OSError, ValueError, KeyError, TypeError):
        return False
