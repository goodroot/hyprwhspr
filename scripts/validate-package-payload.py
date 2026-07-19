#!/usr/bin/env python3
"""Validate that a staged hyprwhspr package contains every dependency plan file."""

import argparse
import ast
from pathlib import Path


class PayloadError(RuntimeError):
    pass


def _plan_manifests(installed_root: Path):
    source = installed_root / "lib" / "src" / "dependency_plan.py"
    if not source.is_file():
        raise PayloadError(f"missing planner: {source}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "PLAN_SPECS"
                for target in node.targets):
            specs = ast.literal_eval(node.value)
            return [installed_root / value[0] for value in specs.values()]
    raise PayloadError(f"PLAN_SPECS not found in {source}")


def _directive(raw, short, long):
    line = raw.split("#", 1)[0].strip()
    parts = line.split()
    if parts and parts[0] in (short, long):
        if len(parts) != 2:
            raise PayloadError(f"invalid {long} directive: {raw.strip()}")
        return parts[1]
    if line.startswith(long + "="):
        return line.split("=", 1)[1]
    if line.startswith(short) and line != short:
        return line[len(short):]
    return None


def validate(installed_root: Path):
    visited, visiting = set(), set()

    def visit(path):
        path = path.resolve()
        if path in visiting:
            raise PayloadError(f"cyclic manifest include: {path}")
        if path in visited:
            return
        if not path.is_file():
            raise PayloadError(f"missing dependency manifest: {path}")
        visiting.add(path)
        for raw in path.read_text(encoding="utf-8").splitlines():
            for short, long in (("-r", "--requirement"), ("-c", "--constraint")):
                target = _directive(raw, short, long)
                if target:
                    if "://" in target:
                        raise PayloadError(f"remote manifest is not deterministic: {target}")
                    visit(path.parent / target)
        visiting.remove(path)
        visited.add(path)

    for manifest in _plan_manifests(installed_root):
        visit(manifest)
    return visited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="staged /usr/lib/hyprwhspr root")
    args = parser.parse_args()
    files = validate(args.root)
    print(f"validated {len(files)} dependency manifest(s)")


if __name__ == "__main__":
    try:
        main()
    except PayloadError as exc:
        raise SystemExit(f"package payload validation failed: {exc}")
