from __future__ import annotations

from pathlib import Path


def test_internal_modules_do_not_import_facades() -> None:
    """Enforce canonical imports inside src internals.

    Facades are allowed only at package public-entry points.
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

    forbidden_tokens = (
        "from pyceles.core.fields import",
        "import pyceles.core.fields",
        "from .fields import",
        "from pyceles.postprocessing.nearfield import",
        "import pyceles.postprocessing.nearfield",
        "from .nearfield import",
    )

    violations: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        if path in allowed:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if any(token in stripped for token in forbidden_tokens):
                rel = path.relative_to(repo_root)
                violations.append(f"{rel}:{lineno}: {stripped}")

    assert not violations, (
        "Internal modules must import canonical implementations, not facades.\n"
        + "\n".join(violations)
    )
