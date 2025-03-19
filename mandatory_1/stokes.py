from collections.abc import Callable

import basix.ufl
import dolfinx
import numpy as np
import pyvista
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from ufl.core import expr


def visualize_mixed(mixed_function: fem.Function, scale: float = 1.0) -> None:
    """Visualize a mixed function.

    This function is borrowed from the FEniCS tutorial:
    https://jsdokken.com/FEniCS-workshop/src/deep_dive/mixed_problems.html

    Args:
        mixed_function (fem.Function): The mixed function to visualize.
        scale (float, optional): The scale of the glyphs. Defaults to 1.0.
    """
    comm: MPI.Intracomm = mixed_function.function_space.mesh.comm
    if comm.size > 1:
        # Only visualize if running on a single rank
        # Otherwise we need to gather the solution to rank 0
        # before visualizing, which complicates things
        if comm.rank == 0:
            print("Visualization only supported for single rank")
        return

    u_c = mixed_function.sub(0).collapse()
    p_c = mixed_function.sub(1).collapse()

    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u_c.function_space))

    # Pad u to be 3D
    gdim = u_c.function_space.mesh.geometry.dim
    assert len(u_c) == gdim
    u_values = np.zeros((len(u_c.x.array) // gdim, 3), dtype=np.float64)
    u_values[:, :gdim] = u_c.x.array.real.reshape((-1, gdim))

    # Create a point cloud of glyphs
    u_grid["u"] = u_values
    glyphs = u_grid.glyph(orient="u", factor=scale)
    plotter = pyvista.Plotter()
    plotter.add_mesh(u_grid, show_edges=False, show_scalar_bar=False)
    plotter.add_mesh(glyphs)
    plotter.view_xy()
    plotter.show()

    p_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(p_c.function_space))
    p_grid.point_data["p"] = p_c.x.array
    plotter_p = pyvista.Plotter()
    plotter_p.add_mesh(p_grid, show_edges=False)
    plotter_p.view_xy()
    plotter_p.show()


def p_boundary(x: np.ndarray) -> np.ndarray:
    return np.isclose(x[0], 1.0)


def setup_mesh(M: int) -> dolfinx.mesh.Mesh:
    """Setup a unit square mesh.

    Args:
        M (int): Number of cells in each direction.

    Returns:
        dolfinx.mesh.Mesh: The mesh.
    """
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, M, M, ghost_mode=dolfinx.cpp.mesh.GhostMode.none
    )
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)

    return mesh


def setup_function_space(
    mesh: dolfinx.mesh.Mesh, u_dim: int, p_dim: int, element_type: str = "Lagrange"
) -> fem.FunctionSpace:
    """Setup the mixed function space.

    Args:
        mesh (dolfinx.mesh.Mesh): The mesh.
        u_dim (int): The polynomial degree of the velocity space.
        p_dim (int): The polynomial degree of the pressure space.
        element_type (str, optional): The type of element. Defaults to "Lagrange".

    Returns:
        fem.FunctionSpace: The mixed function space.
    """
    el_u = basix.ufl.element(
        element_type, mesh.basix_cell(), u_dim, shape=(mesh.geometry.dim,)
    )
    el_p = basix.ufl.element(element_type, mesh.basix_cell(), p_dim)
    el_mixed = basix.ufl.mixed_element([el_u, el_p])

    W = fem.functionspace(mesh, el_mixed)

    return W


def setup_bcs(
    W: fem.FunctionSpace,
    mesh: dolfinx.mesh.Mesh,
    boundary: Callable[[np.ndarray], np.ndarray],
    u_analytical: Callable[[np.ndarray], np.ndarray],
) -> list[fem.DirichletBC]:
    """Setup the boundary conditions.

    Args:
        W (fem.FunctionSpace):
            The mixed function space.
        mesh (dolfinx.mesh.Mesh):
            The mesh.
        boundary (Callable[[np.ndarray], np.ndarray]):
            The boundary condition function.
        u_analytical (Callable[[np.ndarray], np.ndarray]):
            The analytical solution for the velocity at the boundary.

    Returns:
        list[fem.DirichletBC]: The list of Dirichlet boundary conditions.
    """
    fdim = mesh.topology.dim - 1
    W0 = W.sub(0)
    V, _ = W0.collapse()

    bc_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, boundary)
    bc_dofs = fem.locate_dofs_topological((W0, V), fdim, bc_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(u_analytical)
    bc_u = fem.dirichletbc(u_bc, bc_dofs, W0)

    return [bc_u]


def setup_problem(
    W: fem.FunctionSpace, f: ufl.Coefficient, h: ufl.Coefficient
) -> tuple[ufl.Form, ufl.Form]:
    r"""Setup the Stokes problem.

    Here, we consider the problem
        -\delta u - \nabla p = f in \Omega,
        \nabla \cdot u = 0 in \Omega,
        u = u_{\text{analytical}} on \partial \Omega_D,
        \frac{\partial u}{\partial n} - pn = h on \partial \Omega_N.

    Args:
        W (fem.FunctionSpace): The mixed function space.
        f (ufl.Coefficient): The right-hand side.

    Returns:
        tuple[ufl.Form, ufl.Form]: The bilinear and linear forms.
    """
    mesh = W.mesh
    ds = ufl.Measure("ds", domain=mesh)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    # f = -ufl.div(ufl.grad(u)) - ufl.grad(p)
    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    F += ufl.inner(ufl.div(u), q) * ufl.dx
    F += ufl.inner(ufl.div(v), p) * ufl.dx
    F -= ufl.inner(f, v) * ufl.dx
    F -= ufl.inner(h, v) * ds

    a, L = ufl.system(F)

    return a, L


def compute_solution(
    M: int,
    u_dim: int,
    p_dim: int,
    u_analytical: Callable[[np.ndarray], np.ndarray],
    u_boundary: Callable[[np.ndarray], np.ndarray],
    f: ufl.Form | None = None,
    mesh: dolfinx.mesh.Mesh | None = None,
) -> fem.Function:
    if mesh is None:
        mesh = setup_mesh(M)
    W = setup_function_space(mesh, u_dim, p_dim)
    bcs = setup_bcs(W, mesh, u_boundary, u_analytical)
    if f is None:
        f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0, 0)))

    a, L = setup_problem(W, f)
    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    wh = problem.solve()
    # wh = solve_problem(a, L, bcs)
    return wh


def solve_stokes(
    mesh: dolfinx.mesh.Mesh,
    W: fem.FunctionSpace,
    u_boundary: Callable[[np.ndarray], np.ndarray],
) -> tuple[fem.Function, fem.Function]:
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)

    u_ufl = ufl.as_vector((ufl.sin(ufl.pi * x[1]), ufl.cos(ufl.pi * x[0])))
    p_ufl = ufl.sin(2 * ufl.pi * x[0])
    f_ufl = -ufl.div(ufl.grad(u_ufl)) - ufl.grad(p_ufl)
    h_ufl = ufl.grad(u_ufl) * n + n * p_ufl

    bcs = setup_bcs(W, mesh, u_boundary, u_analytical)

    a, L = setup_problem(W, f_ufl, h_ufl)
    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    wh = problem.solve()

    # w_ex = fem.Function(W)
    # u_ex, p_ex = w_ex.split()

    # u_ex.interpolate(u_analytical)
    # p_ex.interpolate(p_analytical)

    return wh, (u_ufl, p_ufl)


def compute_errors(
    Ms: list[int],
    u_dim: int,
    p_dim: int,
    u_boundary: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    if MPI.COMM_WORLD.rank == 0:
        print(f"{u_dim=}, {p_dim=}")
    errors = []
    for M in Ms:
        mesh = setup_mesh(M)
        comm = mesh.comm
        W = setup_function_space(mesh, u_dim, p_dim)
        wh, w_ufl = solve_stokes(mesh, W, u_boundary)

        u_h, p_h = wh.split()
        u_h = wh.sub(0).collapse()
        p_h = wh.sub(1).collapse()
        u_ufl, p_ufl = w_ufl

        E = H1_error(u_h, u_ufl) + L2_error(p_h, p_ufl)
        errors.append(E)

        if comm.rank == 0:
            print(f"h = {1 / M:.2e}, Error = {E:.2e}")

    return np.asarray(errors, dtype=dolfinx.default_scalar_type)


def u_analytical(x: np.ndarray) -> np.ndarray:
    values = np.zeros((2, x.shape[1]))
    values[0] = np.sin(np.pi * x[1])
    values[1] = np.cos(np.pi * x[0])
    return values


def p_analytical(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x[0])


def raised_error(
    uh: fem.Function,
    u_ex: expr.Expr | Callable[[np.ndarray], np.ndarray],
    degree_raise: int = 3,
) -> fem.Function:
    """Setup the error function, by raising the degree.

    Based on https://jsdokken.com/dolfinx-tutorial/chapter4/convergence.html

    Args:
        uh (dolfinx.fem.Function): The numerical solution.
        u_ex (ufl.Coefficient | Callable[[np.ndarray], np.ndarray]): The exact solution.
        degree_raise (int, optional): The degree raise for the space. Defaults to 3.

    Returns:
        dolfinx.fem.Function: The error function.
    """
    org_W = uh.function_space
    degree = org_W.ufl_element().degree
    family = org_W.ufl_element().family_name
    shape = org_W.value_shape
    mesh = org_W.mesh
    W = fem.functionspace(mesh, (family, degree + degree_raise, shape))

    u_W = fem.Function(W)
    u_W.interpolate(uh)

    u_ex_W = fem.Function(W)
    if not isinstance(u_ex, expr.Expr):
        u_ex_W.interpolate(u_ex)
    else:
        u_expr = fem.Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)

    e_W = fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    return e_W


def L2_error(
    uh: fem.Function,
    u_ex: expr.Expr | Callable[[np.ndarray], np.ndarray],
    degree_raise: int = 3,
    dOmega: ufl.Measure = ufl.dx,
) -> float:
    """Compute the L2 error.

    Args:
        uh (dolfinx.fem.Function): The numerical solution.
        u_ex (ufl.Coefficient | Callable[[np.ndarray], np.ndarray]): The exact solution.
        degree_raise (int, optional): The degree raise for the space. Defaults to 3.
        dOmega (ufl.Measure, optional): The measure for the domain. Defaults to ufl.dx.

    Returns:
        float: The L2 error.
    """
    comm: MPI.Comm = uh.function_space.mesh.comm
    e_W = raised_error(uh, u_ex, degree_raise)
    error = fem.form(ufl.inner(e_W, e_W) * dOmega)
    E = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))
    return E


def H1_error(
    uh: fem.Function,
    u_ex: expr.Expr | Callable[[np.ndarray], np.ndarray],
    degree_raise: int = 3,
    dOmega: ufl.Measure = ufl.dx,
) -> float:
    """Compute the H1 error.

    Args:
        uh (dolfinx.fem.Function): The numerical solution.
        u_ex (ufl.Coefficient | Callable[[np.ndarray], np.ndarray]): The exact solution.
        degree_raise (int, optional): The degree raise for the space. Defaults to 3.
        dOmega (ufl.Measure, optional): The measure for the domain. Defaults to ufl.dx.

    Returns:
        float: The H1 error.
    """
    comm: MPI.Comm = uh.function_space.mesh.comm
    e_W = raised_error(uh, u_ex, degree_raise)
    g_e_W = ufl.grad(e_W)
    error = fem.form(ufl.inner(g_e_W, g_e_W) * dOmega)
    E = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))
    return E


if __name__ == "__main__":

    def u_boundary(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)

    Ms = [2, 4, 8, 16, 32, 64, 128]
    Ms = [4, 8, 16, 32, 64]
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
        import matplotlib.pyplot as plt

        fig = plt.figure()

        for (u_dim, p_dim), rate in zip(up_dim, rates, strict=True):
            plt.plot(rate, label=f"$P_{u_dim}$–$P_{p_dim}$")

        plt.xticks(
            range(len(Ms) - 1), [f"$N_{i + 1} / N_{i}$" for i in range(len(Ms) - 1)]
        )
        plt.grid()
        plt.ylabel("$r$")
        plt.title("Order of convergence for the Stokes problem")
        plt.legend()
        plt.savefig("stokes_convergence.pdf", bbox_inches="tight")
