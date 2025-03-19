import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from stokes import (
    compute_errors,
    setup_function_space,
    setup_mesh,
    solve_stokes,
    visualize_mixed,
)


def visualize_error(wh: dolfinx.fem.Function, w_ex: dolfinx.fem.Function) -> None:
    W = wh.function_space
    diff = dolfinx.fem.Function(W)
    diff.x.array[:] = wh.x.array - w_ex.x.array

    visualize_mixed(diff, 0.1)
    visualize_mixed(wh, 0.1)
    visualize_mixed(w_ex, 0.1)


def u_boundary(x: np.ndarray) -> np.ndarray:
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 1.0)


def compute_convergence_of_approx() -> None:
    Ms = [4, 8, 16, 32, 64, 128]
    hs = np.asarray([1 / M for M in Ms])
    up_dim = ((4, 3), (4, 2), (3, 2), (3, 1))
    rates = np.zeros((len(up_dim), len(Ms) - 1), dtype=dolfinx.default_scalar_type)

    for i, (u_dim, p_dim) in enumerate(up_dim):
        errors = compute_errors(Ms, u_dim, p_dim, u_boundary)

        rate = np.log(errors[1:] / errors[:-1]) / np.log(hs[1:] / hs[:-1])
        rates[i] = rate
        if MPI.COMM_WORLD.rank == 0:
            print(rate)

    if MPI.COMM_WORLD.rank == 0:
        fig = plt.figure()

        for (u_dim, p_dim), rate in zip(up_dim, rates, strict=True):
            plt.plot(rate, label=f"$P_{u_dim}$–$P_{p_dim}$")

        plt.xticks(
            range(len(Ms) - 1),
            [f"$N_{i + 1} / N_{i}$" for i in range(len(Ms) - 1)],
        )
        plt.grid()
        plt.ylabel("$r$")
        plt.title("Order of convergence for the Stokes problem")
        plt.legend()
        plt.savefig("wall_shear_conv.pdf", bbox_inches="tight")
        plt.close(fig)


def compute_shear(wh: dolfinx.fem.Function, w_ex: dolfinx.fem.Function) -> float:
    W = wh.function_space
    mesh = W.mesh
    comm: MPI.Comm = mesh.comm
    # ds = ufl.Measure("ds", domain=mesh)
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        mesh,
        mesh.topology.dim - 1,
        lambda x: np.isclose(x[0], 0.0),
    )
    mt = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        boundary_facets,
        np.full(boundary_facets.size, 0, dtype=np.int32),
    )
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
    x = ufl.SpatialCoordinate(mesh)
    # t = ufl.as_vector([0, 1])
    n = ufl.FacetNormal(mesh)
    t = ufl.as_vector([n[1], -n[0]])

    u_h, _ = wh.split()
    u_ex, _ = w_ex.split()
    u_ufl = ufl.as_vector((ufl.sin(ufl.pi * x[1]), ufl.cos(ufl.pi * x[0])))

    # shear_h = ufl.dot(ufl.grad(u_h), t)
    # shear_ex = ufl.dot(ufl.grad(u_ex), t)

    # Matrix multiplication is overloaded to "*" in UFL
    shear_h = ufl.grad(u_h) * t
    shear_ex = ufl.grad(u_ex) * t
    shear_ufl = ufl.grad(u_ufl) * t

    # shear_h = ufl.dot(ufl.grad(u_h), t)
    # shear_ufl = ufl.dot(ufl.grad(u_ufl), t)

    e = shear_ufl - shear_h
    error = fem.form(ufl.inner(e, e) * ds(0))
    # error = fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ds(0))
    E = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))
    # print(E)

    return E


def compute_all_shear(Ms: list[int], u_dim: int, p_dim: int, u_boundary) -> np.ndarray:
    errors = np.zeros(len(Ms), dtype=dolfinx.default_scalar_type)

    for i, M in enumerate(Ms):
        mesh = setup_mesh(M)
        W = setup_function_space(mesh, u_dim, p_dim)
        wh, w_ex = solve_stokes(mesh, W, u_boundary)
        errors[i] = compute_shear(wh, w_ex)

    return errors


if __name__ == "__main__":
    Ms = [4, 8, 16, 32, 64]
    hs = np.asarray([1 / M for M in Ms])
    up_dim = (
        (4, 3),
        # (4, 2),
        (3, 2),
        # (3, 1),
        (2, 1),
    )
    rates = np.zeros((len(up_dim), len(Ms) - 1), dtype=dolfinx.default_scalar_type)

    for i, (u_dim, p_dim) in enumerate(up_dim):
        errors = compute_all_shear(Ms, u_dim, p_dim, u_boundary)

        rate = np.log(errors[1:] / errors[:-1]) / np.log(hs[1:] / hs[:-1])
        print(rate)
        rates[i] = rate

    plt.figure()
    for (u_dim, p_dim), rate in zip(up_dim, rates, strict=True):
        plt.plot(rate, label=f"$P_{u_dim}$–$P_{p_dim}$")

    plt.xticks(
        range(len(Ms) - 1),
        [f"$N_{i + 1} / N_{i}$" for i in range(len(Ms) - 1)],
    )
    plt.grid()
    plt.ylabel("$r$")
    plt.title("Order of convergence for the wall shear stress")
    plt.legend()
    plt.savefig("wall_shear.pdf", bbox_inches="tight")
