from collections.abc import Callable

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
from ufl.core import expr


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


def raised_shear(
    uh: fem.Function,
    u_ex: expr.Expr | Callable[[np.ndarray], np.ndarray],
    degree_raise: int = 3,
):
    org_W = uh.function_space
    degree = org_W.ufl_element().degree
    family = org_W.ufl_element().family_name
    shape = org_W.value_shape
    mesh = org_W.mesh
    comm: MPI.Comm = mesh.comm
    W = fem.functionspace(mesh, (family, degree + degree_raise, shape))

    u_W = fem.Function(W)
    u_W.interpolate(uh)

    u_ex_W = fem.Function(W)
    if isinstance(u_ex, expr.Expr):
        u_expr = fem.Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # t = ufl.as_vector([0, 1])
    # Variable tangential direction, in order to switch boundary
    n = ufl.FacetNormal(mesh)
    t = ufl.as_vector([n[1], -n[0]])

    shear_h = ufl.dot(ufl.grad(u_W), t)
    shear_ex = ufl.dot(ufl.grad(u_ex_W), t)

    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        mesh,
        mesh.topology.dim - 1,
        lambda x: np.isclose(x[0], 0.0),  # x = 0 Dirichlet boundary
        # lambda x: np.isclose(x[1], 0.0), # y = 0 Neumann boundary
    )
    mt = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        boundary_facets,
        np.full(boundary_facets.size, 0, dtype=np.int32),
    )
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)

    e = shear_ex - shear_h
    error = fem.form(ufl.inner(e, e) * ds(0))
    E = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))

    return E


def compute_all_shear(Ms: list[int], u_dim: int, p_dim: int, u_boundary) -> np.ndarray:
    errors = np.zeros(len(Ms), dtype=dolfinx.default_scalar_type)

    for i, M in enumerate(Ms):
        mesh = setup_mesh(M)
        W = setup_function_space(mesh, u_dim, p_dim)
        wh, w_ufl = solve_stokes(mesh, W, u_boundary)
        u_h = wh.sub(0).collapse()
        u_ufl, _ = w_ufl
        errors[i] = raised_shear(u_h, u_ufl)

    return errors


if __name__ == "__main__":
    Ms = [4, 8, 16, 32, 64]
    hs = np.asarray([1 / M for M in Ms])
    up_dim = (
        (5, 4),
        (4, 3),
        (4, 2),
        (3, 2),
        (3, 1),
        (2, 1),
    )
    dash_dot = [":", "-"]
    rates = np.zeros((len(up_dim), len(Ms) - 1), dtype=dolfinx.default_scalar_type)

    for i, (u_dim, p_dim) in enumerate(up_dim):
        errors = compute_all_shear(Ms, u_dim, p_dim, u_boundary)

        rate = np.log(errors[1:] / errors[:-1]) / np.log(hs[1:] / hs[:-1])
        print(rate)
        rates[i] = rate

    plt.figure()
    for i, ((u_dim, p_dim), rate) in enumerate(zip(up_dim, rates, strict=True)):
        plt.plot(rate, dash_dot[i % 2], label=f"$P_{u_dim}$–$P_{p_dim}$")

    plt.xticks(
        range(len(Ms) - 1),
        [f"$N_{i + 1} / N_{i}$" for i in range(len(Ms) - 1)],
    )
    plt.grid()
    plt.ylabel("$r$")
    plt.title("Order of convergence for the wall shear stress")
    plt.legend()
    plt.savefig("wall_shear.pdf", bbox_inches="tight")
