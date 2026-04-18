from __future__ import annotations

import ast
from pathlib import Path


def test_codebase_uses_particle_only_simulation_constructor() -> None:
    """Guard against reintroducing legacy array kwargs in Simulation calls.

    Rationale:
    - geometry is now canonical through explicit particle descriptors,
    - allowing `positions/radii/n_particle` call patterns to creep back in
      would fragment API semantics and conflict with mixed particle families.
    """
    roots = [Path("src"), Path("tests"), Path("examples")]
    forbidden = {"positions", "radii", "n_particle"}
    violations: list[str] = []

    for root in roots:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                is_sim_call = (
                    isinstance(node.func, ast.Name) and node.func.id == "Simulation"
                ) or (isinstance(node.func, ast.Attribute) and node.func.attr == "Simulation")
                if not is_sim_call:
                    continue
                kws = {kw.arg for kw in node.keywords if kw.arg is not None}
                bad = sorted(kws.intersection(forbidden))
                if bad:
                    rel = path.as_posix()
                    violations.append(f"{rel}:{node.lineno}: forbidden kwargs {bad}")

    assert not violations, "Use `Simulation(..., particles=[...])` only.\n" + "\n".join(violations)
