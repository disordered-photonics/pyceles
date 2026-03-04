from __future__ import annotations

import ast
from pathlib import Path


def _module_name_from_path(src_root: Path, path: Path) -> str:
    """Return fully qualified module name for one source file under `src/pyceles`.

    We derive this from file paths so relative imports (``from .x import y``)
    can be resolved semantically, without relying on fragile string matching.
    """
    rel = path.relative_to(src_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        mod_parts = ["pyceles", *parts[:-1]]
    else:
        mod_parts = ["pyceles", *parts[:-1], Path(parts[-1]).stem]
    return ".".join(mod_parts)


def _resolve_from_module(current_module: str, node: ast.ImportFrom) -> str | None:
    """Resolve absolute module path for ``from ... import ...`` nodes.

    Rationale:
    - token-based checks miss many legal syntactic forms (aliases, multiline),
    - AST gives us normalized import nodes,
    - but we still need to resolve relative import levels to compare against
      canonical forbidden facade module names.
    """
    if node.level == 0:
        return node.module
    package_parts = current_module.rsplit(".", 1)[0].split(".")
    drop = int(node.level) - 1
    if drop > len(package_parts):
        return None
    anchor = package_parts[: len(package_parts) - drop]
    if node.module:
        return ".".join([*anchor, node.module])
    return ".".join(anchor)


def _is_forbidden_module(name: str | None, forbidden: set[str]) -> bool:
    """Return True when `name` is one of forbidden facades or their children."""
    if not name:
        return False
    return any(name == base or name.startswith(base + ".") for base in forbidden)


def test_internal_modules_do_not_import_facades() -> None:
    """Enforce canonical imports inside src internals.

    Facades are allowed only at package public-entry points.

    Why this exists
    ---------------
    `pyceles.core.fields` and `pyceles.postprocessing.nearfield` are public
    compatibility facades. Internal modules should import canonical
    implementations directly to avoid accidental layering regressions.

    Why AST (instead of token scanning)
    -----------------------------------
    String-token guards are brittle and can be bypassed by formatting variants.
    Parsing imports via `ast` makes this policy robust to aliases and multiline
    import statements while keeping the test dependency-free (stdlib only).
    """

    src_root: Path | None = None
    repo_root: Path | None = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src" / "pyceles"
        if candidate.exists():
            repo_root = parent
            src_root = candidate
            break
    if repo_root is None or src_root is None:
        raise AssertionError("Could not locate repository root containing src/pyceles.")

    allowed = {
        src_root / "__init__.py",
        src_root / "core" / "__init__.py",
        src_root / "core" / "fields.py",
        src_root / "postprocessing" / "__init__.py",
        src_root / "postprocessing" / "nearfield.py",
    }

    forbidden_facades = {
        "pyceles.core.fields",
        "pyceles.postprocessing.nearfield",
    }

    violations: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        if path in allowed:
            continue
        rel = path.relative_to(repo_root)
        current_module = _module_name_from_path(src_root, path)
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden_module(alias.name, forbidden_facades):
                        violations.append(f"{rel}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                resolved = _resolve_from_module(current_module, node)
                # Catch direct facade imports:
                #   from pyceles.core.fields import ...
                #   from .fields import ...   (when resolved under pyceles.core)
                if _is_forbidden_module(resolved, forbidden_facades):
                    mod_txt = "" if resolved is None else resolved
                    violations.append(f"{rel}:{node.lineno}: from {mod_txt} import ...")
                    continue
                if resolved is not None:
                    # Catch module imports re-exported through package imports:
                    #   from pyceles.core import fields
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        imported_module = f"{resolved}.{alias.name}"
                        if _is_forbidden_module(imported_module, forbidden_facades):
                            violations.append(
                                f"{rel}:{node.lineno}: from {resolved} import {alias.name}"
                            )

    assert not violations, (
        "Internal modules must import canonical implementations, not facades.\n"
        + "\n".join(violations)
    )
