"""Deterministic parser and renderer for local pip manifest graphs."""

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import re
from typing import Callable, Iterable, Optional


class EdgeKind(Enum):
    REQUIREMENT = "requirement"
    CONSTRAINT = "constraint"


@dataclass(frozen=True)
class ManifestEdge:
    kind: EdgeKind
    source: Path
    target: Path
    line_number: int


@dataclass(frozen=True)
class ManifestGraph:
    root: Path
    manifests: tuple[Path, ...]
    edges: tuple[ManifestEdge, ...]


def option_argument(raw: str, short: str, long: str, error: Callable[[str], Exception]) -> Optional[str]:
    line = raw.split("#", 1)[0].strip()
    parts = line.split()
    if parts and parts[0] in (short, long):
        if len(parts) != 2:
            raise error(f"Invalid {long} directive: {raw.strip()}")
        return parts[1]
    if line.startswith(long + "="):
        return line.split("=", 1)[1]
    if line.startswith(short) and line != short:
        return line[len(short):]
    return None


def parse_graph(root: Path, error: Callable[[str], Exception]) -> ManifestGraph:
    ordered, visiting, edges = [], set(), []

    def visit(path: Path):
        path = path.resolve()
        if path in visiting:
            raise error(f"Cyclic dependency manifest include at {path}")
        if path in ordered:
            return
        if not path.is_file():
            raise error(f"Dependency manifest is missing: {path}. Reinstall hyprwhspr; the package payload is incomplete.")
        visiting.add(path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise error(f"Cannot read dependency manifest {path}: {exc}") from exc
        for number, raw in enumerate(lines, 1):
            for kind, short, long in (
                    (EdgeKind.REQUIREMENT, "-r", "--requirement"),
                    (EdgeKind.CONSTRAINT, "-c", "--constraint")):
                target = option_argument(raw, short, long, error)
                if target is None:
                    continue
                if "://" in target:
                    raise error(
                        f"Remote dependency manifest {target!r} in {path}:{number} is not supported; "
                        "vendor it as a local package manifest so installs can be fingerprinted"
                    )
                target_path = (path.parent / target).resolve()
                edges.append(ManifestEdge(kind, path, target_path, number))
                visit(target_path)
        visiting.remove(path)
        ordered.append(path)

    visit(root)
    return ManifestGraph(root.resolve(), tuple(ordered), tuple(edges))


def fingerprint(manifests: Iterable[Path]) -> str:
    manifests = tuple(Path(manifest).resolve() for manifest in manifests)
    if not manifests:
        return hashlib.sha256().hexdigest()
    # parse_graph returns the root last.  Anchor identities at its directory so
    # two included files with the same basename remain distinguishable.
    root_dir = manifests[-1].parent
    digest = hashlib.sha256()
    for manifest in manifests:
        try:
            identity = manifest.relative_to(root_dir).as_posix()
        except ValueError:
            identity = Path("..") / Path(*manifest.parts[1:])
            identity = identity.as_posix()
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(manifest.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def package_name(line: str) -> str:
    match = re.match(r"^([a-z0-9][-a-z0-9_.]*)", line.strip().lower())
    return canonical_name(match.group(1)) if match else ""


def render_filtered(root: Path, output, skipped: Iterable[str], error: Callable[[str], Exception]):
    """Expand requirement edges, retaining constraints as absolute references."""
    graph = parse_graph(root, error)  # full preflight before writing anything
    requirement_targets = {
        (edge.source, edge.line_number): edge.target
        for edge in graph.edges if edge.kind is EdgeKind.REQUIREMENT
    }
    constraint_targets = {
        (edge.source, edge.line_number): edge.target
        for edge in graph.edges if edge.kind is EdgeKind.CONSTRAINT
    }
    skipped = {canonical_name(name) for name in skipped}

    def render(path: Path):
        path = path.resolve()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            key = (path, number)
            if key in requirement_targets:
                render(requirement_targets[key])
            elif key in constraint_targets:
                output.write(f"--constraint {constraint_targets[key]}\n")
            elif package_name(line) not in skipped:
                output.write(line + "\n")
    render(graph.root)
