from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI


def setup_mesh(N: int) -> mesh.Mesh:
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, N)

    return domain


def setup_function_space(domain: mesh.Mesh, p: int) -> fem.FunctionSpace:
    V = fem.functionspace(domain, ("Lagrange", p))

    return V


def setup_boundary_conditions(V: fem.FunctionSpace) -> fem.DirichletBC:
    uD = fem.Function(V)
    uD.interpolate(lambda x: 0 * x[0])

    tdim = V.mesh.topology.dim
    fdim = tdim - 1
    V.mesh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(V.mesh.topology)

    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    return bc


def setup_problem(V: fem.FunctionSpace, bc: fem.DirichletBC, k: float) -> LinearProblem:
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Function(V)
    f.interpolate(lambda x: np.sin(k * np.pi * x[0]))

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )

    return problem


def plot_solution(
    V: fem.FunctionSpace, problem: LinearProblem, ax: plt.Axes, p: int
) -> None:
    uh = problem.solve()

    x = V.tabulate_dof_coordinates()
    x_order = np.argsort(x[:, 0])
    ax.plot(x[x_order, 0], uh.x.array[x_order], label=f"$p={p}$")


def main(N: int = 10, show: bool = False) -> None:
    POLYNOMIAL_DEGREES = [1, 2, 3]
    K_VALUES = [1, 10]

    fig, ax = plt.subplots(1, len(K_VALUES))

    for i, k in enumerate(K_VALUES):
        for p in POLYNOMIAL_DEGREES:
            print(f"Running with {k=}, {p=}, {N=}")
            domain = setup_mesh(N)
            V = setup_function_space(domain, p)
            bc = setup_boundary_conditions(V)
            problem = setup_problem(V, bc, k)
            plot_solution(V, problem, ax[i], p)

    for a, k in zip(ax, K_VALUES):
        a.legend()
        a.set_xlabel("$x$")
        a.set_ylabel("$u_h$")
        a.set_title(f"$k={k}$")

    fig.suptitle(f"Approximation of $u$ with different polynomial degrees, $N = {N}$")
    fig.tight_layout()
    savedir = Path(__file__).parent
    plt.savefig(savedir / f"exercise_1_4_{N=}.pdf")
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    for N in [5, 10, 20, 40]:
        main(N)
