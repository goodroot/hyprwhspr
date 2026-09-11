"""Ownership receipts for existing integration editors, without desktop imports."""
from contextlib import contextmanager
import difflib
import hashlib
import os
from pathlib import Path

try:
    from .managed_install import Installation, atomic_text, digest
except ImportError:
    from managed_install import Installation, atomic_text, digest


# Present only in a widget this installation has already patched. Migration keys
# on it so an already-managed widget is never re-scanned as a legacy reference.
MANAGED_WIDGET_MARKER = '-- hyprwhspr managed tray shim'


def remove_entry(path, entry):
    """Return False when user edits make safe removal ambiguous."""
    path = Path(path)
    if path.name in ('hyprwhspr-module.jsonc', 'hyprwhspr-style.css'):
        for shared in ('config.jsonc', 'style.css'):
            config = path.parent / shared
            try:
                referenced = config.exists() and path.name in config.read_text(encoding='utf-8')
            except (OSError, UnicodeError) as exc:
                print(f'Preserved {path}: could not inspect {config}: {exc}')
                return False
            if referenced:
                print(f'Preserved {path}: still referenced by {config}')
                return False
    if not path.exists() and not path.is_symlink():
        return True
    if 'target' in entry:
        if not path.is_symlink() or os.readlink(path) != entry['target']:
            return False
        path.unlink()
        return True
    if path.is_symlink():
        return False
    blocks = entry.get('insertions', []) if 'restore_sha256' in entry else []
    try:
        content = path.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as exc:
        print(f'Preserved unreadable integration: {path}: {exc}')
        return False
    if blocks and all(content.count(block) == 1 for block in blocks):
        for block in blocks:
            content = content.replace(block, '', 1)
        atomic_text(path, content, path.stat().st_mode & 0o777)
        return True
    if entry.get('original') is not None:
        # A latest-write digest can include user edits from an intervening setup.
        # Only the first installation output authorizes a wholesale restore.
        if digest(path) != entry.get('restore_sha256'):
            return False
        atomic_text(path, entry['original'], path.stat().st_mode & 0o777)
    elif digest(path) == entry['sha256'] and entry.get('restore_sha256', entry['sha256']) == entry['sha256']:
        # A file this installation created is only ours to delete while it still
        # holds installation-authored content. Once a user edit intervenes,
        # restore_sha256 stops advancing and the file is preserved instead.
        path.unlink()
    else:
        return False
    return True


def refresh_insertion(block, before, after):
    """Follow edits wholly inside an owned block; never absorb adjacent user text."""
    if after.count(block) == 1 or before.count(block) != 1:
        return block
    start = before.index(block)
    end = start + len(block)
    pieces = []
    for op, i, j, k, l in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
        if op == 'insert':
            if start < i < end:
                pieces.append(after[k:l])
        elif i < end and j > start:
            if op == 'equal':
                pieces.append(after[k + max(start, i) - i:k + min(end, j) - i])
            elif i >= start and j <= end:
                pieces.append(after[k:l])
            else:
                return block
    updated = ''.join(pieces)
    return updated if updated and after.count(updated) == 1 else block


@contextmanager
def edit_files(paths, shared=(), mode='install', removal=None, allow_shared_preservation=False):
    """Wrap a legacy editor and record actual changes, including partial failure.

    Shared files are restored only when unchanged, or when additions are exact,
    unique text blocks. Dedicated application files must be absent or owned.
    """
    install = Installation()
    receipt = install.read_receipt()
    files = receipt.setdefault('files', {})
    paths = [Path(p) for p in paths]
    shared = {Path(p) for p in shared}
    resolved = {}
    for path in paths:
        if path in shared and path.exists():
            try:
                target = path.resolve(strict=True)
                if not target.is_file():
                    raise OSError('shared target is not a regular file')
                resolved[path] = target
            except (OSError, RuntimeError) as exc:
                print(f'Skipping integration; invalid shared symlink: {path}: {exc}')
                yield False
                return
    paths = list(dict.fromkeys(resolved.get(path, path) for path in paths))
    shared = {resolved.get(path, path) for path in shared}
    if mode == 'remove':
        complete = True
        with install.receipts() as receipt:
            files = receipt.setdefault('files', {})
            for path in sorted(paths, key=lambda p: p.name.startswith('hyprwhspr-')):
                entry = files.get(str(path))
                try:
                    if entry and remove_entry(path, entry):
                        del files[str(path)]
                        print(f'Removed owned integration: {path}')
                    elif path.exists() or path.is_symlink():
                        complete = complete and allow_shared_preservation and path in shared
                        print(f'Preserved modified or unowned integration: {path}')
                except (OSError, UnicodeError) as exc:
                    complete = complete and allow_shared_preservation and path in shared
                    print(f'Could not remove integration; ownership retained: {path}: {exc}')
        if removal is not None:
            removal['complete'] = complete
        yield False
        return
    before = {}
    for path in paths:
        if path.is_symlink():
            print(f'Skipping integration; symlink preserved: {path}')
            yield False
            return
        if path.exists():
            try:
                before[path] = path.read_text(encoding='utf-8')
            except (OSError, UnicodeError) as exc:
                print(f'Skipping integration; file preserved because it could not be read: {path}: {exc}')
                yield False
                return
            entry = files.get(str(path))
            if path not in shared and (not entry or digest(path) != entry.get('sha256')):
                print(f'Skipping integration; modified or unowned file preserved: {path}')
                yield False
                return
        else:
            before[path] = None
    try:
        yield True
    finally:
        with install.receipts() as receipt:
            files = receipt.setdefault('files', {})
            for path, original in before.items():
                if not path.exists():
                    continue
                try:
                    after = path.read_text(encoding='utf-8')
                    fingerprint = digest(path)
                except (OSError, UnicodeError) as exc:
                    print(f'Could not verify edited integration; previous ownership retained: {path}: {exc}')
                    continue
                if after == original:
                    continue
                previous = files.get(str(path), {})
                baseline = previous.get('original', original)
                entry = {'sha256': fingerprint, 'kind': 'integration', 'original': baseline}
                # restore_sha256 tracks the last content this installation authored.
                # It advances only while no user edit intervened, so a wholesale
                # restore or delete can never act on a file carrying user changes.
                pre = (hashlib.sha256(original.encode('utf-8')).hexdigest()
                       if original is not None else None)
                # A file we created and the user deleted carries no user content, so
                # regenerating it still leaves the whole file installation-authored.
                recreated = original is None and previous.get('original') is None
                untouched = (not previous or recreated or
                             (previous.get('sha256') == pre and
                              previous.get('restore_sha256', previous.get('sha256')) == pre))
                entry['restore_sha256'] = (entry['sha256'] if untouched
                                           else previous.get('restore_sha256'))
                # Compare only this editor's changes, never baseline-to-current:
                # intervening user additions are not installation-owned blocks.
                blocks = list(previous.get('insertions', [])) if 'restore_sha256' in previous else []
                if original is not None:
                    blocks = [refresh_insertion(block, original, after) for block in blocks]
                    old_lines, new_lines = original.splitlines(keepends=True), after.splitlines(keepends=True)
                    operations = difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes()
                    if all(op in ('equal', 'insert') for op, *_ in operations):
                        blocks.extend(''.join(new_lines[j:k]) for op, _, _, j, k in operations if op == 'insert')
                if blocks:
                    entry['insertions'] = list(dict.fromkeys(blocks))
                files[str(path)] = entry


def stable_hyprland_content(content, data, roots=()):
    """Replace only an exact tray-record command, preserving the user's key chord."""
    import re
    import shlex
    data = Path(data)
    suffix = ('config', 'hyprland', 'hyprwhspr-tray.sh')
    known = {Path(root).joinpath(*suffix) for root in roots}
    stable = shlex.quote(str(data / 'launcher')) + ' --managed-tray record'
    def replace(match):
        try:
            args = shlex.split(match.group(2))
        except ValueError:
            return match.group(0)
        if len(args) != 2 or args[1] != 'record':
            raw = match.group(2).strip()
            if not raw.endswith('/config/hyprland/hyprwhspr-tray.sh record'):
                return match.group(0)
            # Older generated bindings did not quote XDG paths containing spaces.
            args = [raw[:-len(' record')], 'record']
        command = Path(args[0])
        owned = command in known
        try:
            parts = command.relative_to(data / 'releases').parts
            owned |= len(parts) == 4 and re.fullmatch(r'[A-Za-z0-9_-]+', parts[0]) is not None and parts[1:] == suffix
        except ValueError:
            pass
        return match.group(1) + stable if owned else match.group(0)
    return re.sub(r'(?m)^(\s*bind\w*\s*=.*?\bexec,\s*)([^\n]+)', replace, content)

def managed_widget_content(content, tray):
    """Patch the packaged Luau widget, rejecting drift before writing it."""
    quoted = '"' + ''.join(('\\' + char) if char in ('\\', '"') else
                           ('\\%03d' % ord(char)) if ord(char) < 32 else char
                           for char in str(tray)) + '"'
    historical = 'local tray = root .. "/config/hyprland/hyprwhspr-tray.sh"'
    if content.count(historical) == 1 and 'local root = noctalia.getenv("HYPRWHSPR_ROOT")' in content:
        # Mark this branch too: convergence must not depend on the tray path
        # disappearing, which a stray mention elsewhere in the file would defeat.
        return content.replace(historical, MANAGED_WIDGET_MARKER + '\nlocal tray = ' + quoted, 1)
    # Prefer the managed shim but leave TRAY_REL and the packaged root candidates
    # intact, so a missing shim still resolves through the documented fallbacks
    # instead of probing bare directories.
    required = 'local TRAY_REL = "/config/hyprland/hyprwhspr-tray.sh"'
    marker = 'local candidates = {}'
    for expected in (required, marker):
        if content.count(expected) != 1:
            raise RuntimeError(f'Noctalia widget template changed: expected exactly one {expected!r}')
    replacement = (f'{MANAGED_WIDGET_MARKER}\n'
                   f'  local managed = {quoted}\n'
                   '  if noctalia.fileExists(managed) then\n'
                   '    trayCached = managed\n'
                   '    return managed\n'
                   '  end\n'
                   f'  {marker}')
    return content.replace(marker, replacement, 1)
