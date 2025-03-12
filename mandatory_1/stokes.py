import basix.ufl
import dolfinx
import numpy as np
import pyvista
import ufl
from dolfinx import fem
from mpi4py import MPI


def visualize_mixed(mixed_function: fem.Function, scale: float = 1.0) -> None:
    """Visualize a mixed function.

    This function is borrowed from the FEniCS tutorial:
    https://jsdokken.com/FEniCS-workshop/src/deep_dive/mixed_problems.html

    Args:
        mixed_function (fem.Function): The mixed function to visualize.
        scale (float, optional): The scale of the glyphs. Defaults to 1.0.
    """
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


def u_boundary(x: ufl.SpatialCoordinate) -> np.ndarray:
    return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


def p_boundary(x: ufl.SpatialCoordinate) -> np.ndarray:
    return np.isclose(x[0], 1.0)


M = 6
U_DIM = 2
P_DIM = 1
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, M, M)
tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, tdim)
x = ufl.SpatialCoordinate(mesh)


el_u = basix.ufl.element(
    "Lagrange", mesh.basix_cell(), U_DIM, shape=(mesh.geometry.dim,)
)
el_p = basix.ufl.element("Lagrange", mesh.basix_cell(), P_DIM)
el_mixed = basix.ufl.mixed_element([el_u, el_p])

W = fem.functionspace(mesh, el_mixed)
u, p = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0, 0)))


def p_analytical(x: ufl.SpatialCoordinate):
    return -2 + 2 * x[0]


W0 = W.sub(0)

fdim = tdim - 1
bc_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, u_boundary)
V, _ = W0.collapse()
bc_dofs = fem.locate_dofs_topological((W0, V), fdim, bc_facets)


def u_analytical(x):
    values = np.zeros((2, x.shape[1]))
    # values[0] = x[1] * (1 - x[1])
    values[0] = np.sin(np.pi * x[1])
    values[1] = np.cos(np.pi * x[0])
    return values


u_bc = fem.Function(V)
u_bc.interpolate(u_analytical)
bc_u = fem.dirichletbc(u_bc, bc_dofs, W0)

bcs = [bc_u]

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
a += ufl.inner(ufl.div(u), q) * ufl.dx
a += ufl.div(v) * p * ufl.dx
L = ufl.inner(f, v) * ufl.dx

a_compiled = fem.form(a)
L_compiled = fem.form(L)
A = fem.create_matrix(a_compiled)
b = fem.create_vector(L_compiled)
A_scipy = A.to_scipy()
fem.assemble_matrix(A, a_compiled, bcs=bcs)
fem.assemble_vector(b.array, L_compiled)
fem.apply_lifting(b.array, [a_compiled], [bcs])
b.scatter_reverse(dolfinx.la.InsertMode.add)
[bc.set(b.array) for bc in bcs]

# wh = fem.Function(W)
# uh, ph = ufl.split(wh)

# ds = ufl.Measure("ds", domain=mesh)
# g = fem.Constant(mesh, dolfinx.default_scalar_type((0, 0)))

# f = fem.Constant(mesh, dolfinx.default_scalar_type((0, 0)))

# F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
# F += ufl.inner(p, ufl.div(v)) * ufl.dx
# F += ufl.inner(ufl.div(u), q) * ufl.dx
# F -= ufl.inner(f, v) * ufl.dx

# a, L = ufl.system(F)


def boundary_marker(x):
    return np.isclose(x[0], 0.0)


left_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, boundary_marker)


def top_bottom_marker(x):
    return np.isclose(x[1], 1.0) | np.isclose(x[1], 0.0)


tb_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, top_bottom_marker)


# W0 = W.sub(0)
# V, V_to_W0 = W0.collapse()
# u_inlet_x = fem.Constant(mesh, dolfinx.default_scalar_type(1))
# u_inlet_y = fem.Constant(mesh, dolfinx.default_scalar_type(0))
# dofs_inlet_x = fem.locate_dofs_topological(W0.sub(0), tdim - 1, left_facets)
# dofs_inlet_y = fem.locate_dofs_topological(W0.sub(1), tdim - 1, left_facets)
# bc_inlet_x = fem.dirichletbc(u_inlet_x, dofs_inlet_x, W0.sub(0))
# bc_inlet_y = fem.dirichletbc(u_inlet_y, dofs_inlet_y, W0.sub(1))

# u_wall = fem.Function(V)
# u_wall.x.array[:] = 0
# dofs_wall = fem.locate_dofs_topological((W0, V), tdim - 1, tb_facets)
# bc_wall = fem.dirichletbc(u_wall, dofs_wall, W0)

# a_compiled = fem.form(a)
# L_compiled = fem.form(L)
# A = fem.create_matrix(a_compiled)
# b = fem.create_vector(L_compiled)
# A_scipy = A.to_scipy()
# bcs = [bc_inlet_x, bc_inlet_y, bc_wall]
# fem.assemble_matrix(A, a_compiled, bcs=bcs)

# fem.assemble_vector(b.array, L_compiled)
# fem.apply_lifting(b.array, [a_compiled], [bcs])
# b.scatter_reverse(dolfinx.la.InsertMode.add)
# [bc.set(b.array) for bc in bcs]

import scipy.sparse

A_inv = scipy.sparse.linalg.splu(A_scipy)

wh = fem.Function(W)
wh.x.array[:] = A_inv.solve(b.array)
visualize_mixed(wh, scale=0.1)
