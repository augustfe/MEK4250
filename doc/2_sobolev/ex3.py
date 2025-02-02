import math
from collections.abc import Callable

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI


def bernstein_polynomial(x: ufl.SpatialCoordinate, n: int, i: int) -> float:
    return math.comb(n, i) * x[0] ** i * (1 - x[0]) ** (n - i)


def L2_norm(f: fem.Function) -> float:
    return np.sqrt(fem.assemble_scalar(fem.form(ufl.dot(f, f) * ufl.dx)))


def H1_norm(f: fem.Function) -> float:
    return np.sqrt(
        fem.assemble_scalar(
            fem.form((ufl.dot(f, f) + ufl.dot(ufl.grad(f), ufl.grad(f))) * ufl.dx)
        )
    )


def compute_norms(
    num_cells: int, func: Callable[[float], float]
) -> tuple[float, float]:
    """
    Create a mesh on [0, 1] with `num_cells` cells,
    interpolate a function f(x) in (0, 1),
    and compute its L2 and H1 norms.
    """
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, num_cells)

    V = fem.functionspace(domain, ("Lagrange", 1))

    f = fem.Function(V)
    f.interpolate(func)

    L2 = L2_norm(f)
    H1 = H1_norm(f)

    return L2, H1


def main(func: Callable[[float], float]) -> None:
    # List of number of cells in the mesh
    cell_counts = [10, 100, 1000]

    for N in cell_counts:
        L2_val, H1_val = compute_norms(N, func)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Mesh with {N} cells:")
            print(f"  L2  norm(f) = {L2_val:.6e}")
            print(f"  H1  norm(f) = {H1_val:.6e}")
            print("-" * 40)


if __name__ == "__main__":
    main(lambda x: bernstein_polynomial(x, 10, 5))
