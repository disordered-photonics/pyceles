"""Inspect a published dense ``.tmat.h5`` file without running a solver."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyceles as pcl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmatrix", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--wavelength", type=float, help="exact wavelength in file units")
    selection.add_argument("--wavelength-index", type=int, help="spectral matrix index")
    return parser


def main() -> None:
    args = _parser().parse_args()
    data = pcl.load_tmatrix_h5(
        args.tmatrix,
        wavelength=args.wavelength,
        wavelength_index=args.wavelength_index,
    )
    print(f"file: {args.tmatrix}")
    print(f"matrix: shape={data.t_matrix.shape}, dtype={data.t_matrix.dtype}, lmax={data.lmax}")
    print(
        f"spectrum: axis={data.source_axis!r}, wavelength={data.wavelength!r} "
        f"{data.wavelength_unit or ''}"
    )
    print(f"basis: source={data.source_basis!r}, pyceles=CELES parity")
    print(f"embedding refractive index: {data.embedding_refractive_index!r}")
    print(f"norm: frobenius={float((abs(data.t_matrix) ** 2).sum() ** 0.5):.16g}")
    print(f"max entry: {float(abs(data.t_matrix).max()):.16g}")
    print(f"metadata: embedding={sorted(data.embedding)}, attributes={sorted(data.attributes)}")


if __name__ == "__main__":
    main()
