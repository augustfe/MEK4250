import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from scipy.sparse import linalg as la

N = 32
N = 500
mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)
alpha = dolfinx.default_scalar_type(1e-5)
tdim = mesh.topology.dim
V = fem.functionspace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)

mesh.topology.create_entities(tdim - 1)
mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
bc = fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs, V)
bcs = [bc]


def L1(with_x: bool = False) -> ufl.Form:
    if not with_x:
        return u.dx(0) * v
    return x[0] * u.dx(0) * v


def L2() -> ufl.Form:
    return alpha * ufl.inner(ufl.grad(u), ufl.grad(v))


def L3(with_x: bool = False) -> ufl.Form:
    return L1(with_x=with_x) + L2()


def compute_eigs(which_a: str) -> np.ndarray:
    match which_a:
        case "L1":
            a_term = L1()
        case "L2":
            a_term = L2()
        case "L3":
            a_term = L3()
        case "L1x":
            a_term = L1(with_x=True)
        case "L3x":
            a_term = L3(with_x=True)
        case _:
            raise ValueError("Invalid value for which_a")

    print(f"Computing eigs for {which_a}")
    a_form = fem.form(a_term * ufl.dx)
    m_form = fem.form(u * v * ufl.dx)
    A = fem.assemble_matrix(a_form, bcs=bcs)  # .to_scipy()
    A.scatter_reverse()
    # A.assemble()  # .to_scipy()
    M = fem.assemble_matrix(m_form, bcs=bcs)  # .to_scipy()
    M.scatter_reverse()
    # M.assemble()  # .to_scipy()

    A = A.to_scipy()
    M = M.to_scipy()

    eigs = la.eigs(
        A,
        M=M,
        k=N - 1,
        # k=20,
        return_eigenvectors=False,
        # which="SI",
        maxiter=200000,
    )
    print(eigs)

    # Exclude dirichlet value
    if "L1" in which_a or "L3" in which_a:
        eigs = eigs[1:-3]
    return eigs


def plot_all_eigs(show: bool = False, save: bool = False) -> None:
    _, axs = plt.subplots(2, 3, figsize=(8, 5))
    which_as = ["L1", "L2", "L3", "L1x", "L2", "L3x"]
    for i, which_a in enumerate(which_as):
        eigs = compute_eigs(which_a)
        ax = axs[i // 3, i % 3]
        ax.plot(eigs.real, eigs.imag, "o")
        ax.set_title(which_a)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_box_aspect(1)

    plt.tight_layout()
    if save:
        plt.savefig("eigs.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    plot_all_eigs(show=True, save=False)
