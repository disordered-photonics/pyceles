"""Run the public performance-profile cases in isolated Python processes.

The individual profiling scripts remain useful for targeted work.  This small
orchestrator provides one reproducible command for refreshing the documented
finite and periodic snapshots without sharing CuPy allocator state between
cases.  Solver/preparation cases are kept separate from a small set of
representative postprocessing cases.  Periodic NumPy cache-off is retained as
a one-iteration reference probe: its full solve is
impractical for the 500-particle profile geometry, but its single-iteration
cost remains useful for extrapolation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _case_has_summary(case: dict[str, Any]) -> bool:
    output_dir = Path(case["output_dir"])
    return any(
        (output_dir / filename).is_file()
        for filename in ("profile_summary.json", "profile_periodic_summary.json")
    )


def _case(
    *,
    name: str,
    script: str,
    arguments: list[str],
    output_dir: Path,
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "script": script,
        "arguments": arguments,
        "output_dir": str(output_dir),
        "skipped_reason": skipped_reason,
    }


def _field_arguments(
    args: argparse.Namespace,
    *,
    periodic: bool,
    include_postprocessing: bool,
) -> list[str]:
    if args.skip_farfield and not args.skip_nearfield:
        raise ValueError(
            "--skip-farfield also requires --skip-nearfield because the finite "
            "profile uses the far-field payload for near-field evaluation."
        )
    result: list[str] = []
    if not include_postprocessing:
        result.append("--skip-nearfield")
        if not periodic:
            result.append("--skip-farfield")
        return result
    if args.skip_nearfield:
        result.append("--skip-nearfield")
    if args.skip_farfield and not periodic:
        result.append("--skip-farfield")
    return result


def _build_cases(args: argparse.Namespace, output_root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if args.suite in ("finite", "all"):
        for backend in ("numpy", "cupy"):
            for dtype in ("complex128", "complex64"):
                name = f"finite_{backend}_{dtype}"
                case_dir = output_root / name
                command = [
                    "--operator-backend",
                    backend,
                    "--compute-dtype",
                    dtype,
                    "--accum-dtype",
                    "complex128",
                    "--solver",
                    "bicgstab",
                    "--cache-mode",
                    args.finite_cache_mode,
                    "--out-dir",
                    str(case_dir),
                ]
                representative = backend == "cupy" and dtype == "complex64"
                command.extend(
                    _field_arguments(
                        args,
                        periodic=False,
                        include_postprocessing=(representative and not args.skip_postprocessing),
                    )
                )
                cases.append(
                    _case(
                        name=name,
                        script="profile_pyceles_phases.py",
                        arguments=command,
                        output_dir=case_dir,
                    )
                )

    if args.suite in ("periodic", "all"):
        for backend in ("numpy", "cupy"):
            name = f"periodic_pairwise_{backend}_cache_on"
            case_dir = output_root / name
            representative = backend == "cupy" and not args.skip_postprocessing
            cases.append(
                _case(
                    name=name,
                    script="profile_pyceles_periodic_phases.py",
                    arguments=[
                        "--coupling-backend",
                        "pairwise",
                        "--operator-backend",
                        backend,
                        "--periodic-method",
                        "ewald",
                        "--cache-mode",
                        "on",
                        "--solver",
                        "gmres",
                        "--solver-maxiter",
                        str(args.periodic_maxiter),
                        "--out-dir",
                        str(case_dir),
                        *_field_arguments(
                            args,
                            periodic=True,
                            include_postprocessing=representative,
                        ),
                    ],
                    output_dir=case_dir,
                )
            )

            if not args.skip_periodic_cache_off:
                cache_off_name = f"periodic_pairwise_{backend}_cache_off"
                cache_off_dir = output_root / cache_off_name
                if backend == "numpy":
                    cache_off_name = f"{cache_off_name}_1iter"
                    cache_off_dir = output_root / cache_off_name
                    cases.append(
                        _case(
                            name=cache_off_name,
                            script="profile_pyceles_periodic_phases.py",
                            arguments=[
                                "--coupling-backend",
                                "pairwise",
                                "--operator-backend",
                                backend,
                                "--periodic-method",
                                "ewald",
                                "--cache-mode",
                                "off",
                                "--solver",
                                "gmres",
                                "--solver-maxiter",
                                str(args.numpy_cache_off_maxiter),
                                "--skip-final-residual-check",
                                "--out-dir",
                                str(cache_off_dir),
                                *_field_arguments(
                                    args,
                                    periodic=True,
                                    include_postprocessing=False,
                                ),
                            ],
                            output_dir=cache_off_dir,
                        )
                    )
                else:
                    cases.append(
                        _case(
                            name=cache_off_name,
                            script="profile_pyceles_periodic_phases.py",
                            arguments=[
                                "--coupling-backend",
                                "pairwise",
                                "--operator-backend",
                                backend,
                                "--periodic-method",
                                "ewald",
                                "--cache-mode",
                                "off",
                                "--solver",
                                "gmres",
                                "--solver-maxiter",
                                str(args.periodic_maxiter),
                                "--out-dir",
                                str(cache_off_dir),
                                *_field_arguments(
                                    args,
                                    periodic=True,
                                    include_postprocessing=False,
                                ),
                            ],
                            output_dir=cache_off_dir,
                        )
                    )

            if not args.skip_direct:
                name = f"periodic_direct_{backend}"
                case_dir = output_root / name
                cases.append(
                    _case(
                        name=name,
                        script="profile_pyceles_periodic_phases.py",
                        arguments=[
                            "--coupling-backend",
                            "pairwise",
                            "--operator-backend",
                            backend,
                            "--periodic-method",
                            "ewald",
                            "--cache-mode",
                            "off",
                            "--solver",
                            "direct",
                            "--skip-nearfield",
                            "--skip-final-residual-check",
                            "--out-dir",
                            str(case_dir),
                        ],
                        output_dir=case_dir,
                    )
                )

        for coupling, method, skip_flag in (
            ("mlfmm", "ewald", args.skip_periodized_mlfmm),
            ("pairwise", "rayleigh", args.skip_rayleigh),
        ):
            if skip_flag:
                continue
            method_name = "periodized_mlfmm" if coupling == "mlfmm" else "rayleigh"
            name = f"periodic_{method_name}_cupy"
            case_dir = output_root / name
            cases.append(
                _case(
                    name=name,
                    script="profile_pyceles_periodic_phases.py",
                    arguments=[
                        "--coupling-backend",
                        coupling,
                        "--operator-backend",
                        "cupy",
                        "--periodic-method",
                        method,
                        "--cache-mode",
                        "off",
                        "--solver",
                        "gmres",
                        "--solver-maxiter",
                        str(args.periodic_maxiter),
                        "--out-dir",
                        str(case_dir),
                        *_field_arguments(
                            args,
                            periodic=True,
                            include_postprocessing=False,
                        ),
                    ],
                    output_dir=case_dir,
                )
            )
            name = f"periodic_{method_name}_numpy_1iter"
            case_dir = output_root / name
            cases.append(
                _case(
                    name=name,
                    script="profile_pyceles_periodic_phases.py",
                    arguments=[
                        "--coupling-backend",
                        coupling,
                        "--operator-backend",
                        "numpy",
                        "--periodic-method",
                        method,
                        "--cache-mode",
                        "off",
                        "--solver",
                        "gmres",
                        "--solver-maxiter",
                        str(args.numpy_advanced_maxiter),
                        "--skip-final-residual-check",
                        "--out-dir",
                        str(case_dir),
                        *_field_arguments(
                            args,
                            periodic=True,
                            include_postprocessing=False,
                        ),
                    ],
                    output_dir=case_dir,
                )
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the reproducible public finite and periodic performance profiles."
    )
    parser.add_argument("--suite", choices=("finite", "periodic", "all"), default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/performance_suite"),
        help="Root directory for one subdirectory per profile case.",
    )
    parser.add_argument(
        "--finite-cache-mode",
        choices=("off", "on", "both"),
        default="off",
        help="Translation-cache mode for the finite-cluster profiles.",
    )
    parser.add_argument("--periodic-maxiter", type=int, default=800)
    parser.add_argument(
        "--numpy-cache-off-maxiter",
        type=int,
        default=1,
        help="Iteration count for the slow NumPy periodic cache-off reference probe.",
    )
    parser.add_argument(
        "--numpy-advanced-maxiter",
        type=int,
        default=1,
        help="Iteration count for the slow NumPy Rayleigh/periodized-MLFMM probes.",
    )
    parser.add_argument("--skip-periodic-cache-off", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-rayleigh", action="store_true")
    parser.add_argument("--skip-periodized-mlfmm", action="store_true")
    parser.add_argument(
        "--skip-postprocessing",
        action="store_true",
        help="Skip the representative near/far postprocessing cases.",
    )
    parser.add_argument("--skip-nearfield", action="store_true")
    parser.add_argument("--skip-farfield", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse cases with an existing completed profile summary.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        cases = _build_cases(args, output_root)
    except ValueError as exc:
        parser.error(str(exc))

    manifest: dict[str, Any] = {
        "suite": args.suite,
        "python": sys.executable,
        "output_root": str(output_root),
        "cases": [],
    }
    for case in cases:
        record = dict(case)
        if case["skipped_reason"] is not None:
            print(f"SKIP {case['name']}: {case['skipped_reason']}")
            record["status"] = "skipped"
            manifest["cases"].append(record)
            continue
        if args.reuse_existing and _case_has_summary(case):
            print(f"REUSE {case['name']}: existing profile summary")
            record["status"] = "reused"
            manifest["cases"].append(record)
            continue
        command = [sys.executable, str(root / "examples" / case["script"]), *case["arguments"]]
        record["command"] = command
        print(f"\n=== {case['name']} ===")
        print(subprocess.list2cmdline(command))
        if args.dry_run:
            record["status"] = "planned"
            manifest["cases"].append(record)
            continue
        completed = subprocess.run(command, cwd=root, check=False)
        record["returncode"] = int(completed.returncode)
        record["status"] = "completed" if completed.returncode == 0 else "failed"
        manifest["cases"].append(record)
        if completed.returncode != 0 and not args.continue_on_error:
            manifest["status"] = "failed"
            (output_root / "suite_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            raise SystemExit(completed.returncode)

    manifest["status"] = "planned" if args.dry_run else "completed"
    manifest_path = output_root / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote suite manifest: {manifest_path}")
    if any(case.get("status") == "failed" for case in manifest["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
