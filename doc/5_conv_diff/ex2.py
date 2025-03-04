from collections.abc import Callable

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI


class ConvectionDiffusion:
    def __init__(
        self,
        num_cells: int,
        analytical: Callable[[np.ndarray], np.ndarray],
        mu_value: float = 1e-2,
    ):
        self.num_cells = num_cells
        self.analytical = analytical
        self.order = 1

        self.domain = mesh.create_unit_interval(MPI.COMM_WORLD, self.num_cells)
        self.tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(self.tdim, self.tdim)

        self.V = fem.functionspace(self.domain, ("Lagrange", self.order))
        self.mu = fem.Constant(self.domain, mu_value)
        self.f = fem.Constant(self.domain, 0.0)

        self.h = self.domain.h(self.tdim, np.arange(self.num_cells)).min()

        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

    @property
    def a(self) -> ufl.Form:
        return (-self.u.dx(0) * self.v + self.mu * self.u.dx(0) * self.v.dx(0)) * ufl.dx

    @property
    def L(self) -> ufl.Form:
        return self.f * self.v * ufl.dx

    @staticmethod
    def boundary(x: np.ndarray) -> fem.Function:
        return np.isclose(x[0], 0) | np.isclose(x[0], 1)

    def solve(self) -> np.ndarray:
        dofs_D = fem.locate_dofs_geometrical(self.V, self.boundary)
        u_bc = fem.Function(self.V)
        u_bc.interpolate(self.analytical)
        bc = fem.dirichletbc(u_bc, dofs_D)

        problem = LinearProblem(
            self.a,
            self.L,
            bcs=[bc],
            # petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()

        return uh

    def solve_analytical(self) -> fem.Function:
        u_analytical = fem.Function(self.V)
        u_analytical.interpolate(self.analytical)

        return u_analytical

    def norm(self, u: fem.Function, t: int = 1) -> float:
        integrand = ufl.dot(u, u)
        for _ in range(t):
            u = u.dx(0)
            integrand += ufl.dot(u, u)

        val = fem.assemble_scalar(fem.form(integrand * ufl.dx))
        return np.sqrt(MPI.COMM_WORLD.allreduce(val, op=MPI.SUM))

    def constant_estimate(self) -> float:
        uh = self.solve()
        u_analytical = self.solve_analytical()
        upper = self.norm(u_analytical - uh, t=1)
        lower = self.norm(u_analytical, t=self.order + 1)

        C1 = upper / (lower * (self.h**self.order))
        return C1


if __name__ == "__main__":
    from functools import partial

    import pandas as pd

    mu_values = [1e-1, 1e-2, 1e-3]
    num_cells = [10, 100, 1000, 10000]

    values = np.ndarray((len(num_cells), len(mu_values)))

    def u_exact(x: np.ndarray, mu_value) -> np.ndarray:
        return (np.exp(-x[0] / mu_value) - 1) / (np.exp(-1 / mu_value) - 1)

    for i, mu_value in enumerate(mu_values):
        u_e = partial(u_exact, mu_value=mu_value)

        for j, num_cell in enumerate(num_cells):
            convdiff = ConvectionDiffusion(num_cell, u_e, mu_value)
            C1 = convdiff.constant_estimate()
            values[j, i] = C1

    df = pd.DataFrame(values, index=[1 / N for N in num_cells], columns=mu_values)
    print(df)

    print(df.to_latex())
