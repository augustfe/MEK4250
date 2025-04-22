"""Script for defining the Hermite polynomials on the unit inverval. Exercise 3.2"""

from typing import Literal

import sympy as sp

x = sp.symbols("x")


def dirac(i, j) -> Literal[1] | Literal[0]:
    """Dirac delta function"""
    if i == j:
        return 1
    return 0


def setup_basis(k: int) -> list[sp.Expr]:
    """Setup the basis for the monomials."""
    return [x**i for i in range(k)]


def setup_points(k: int) -> list[sp.Rational]:
    """Setup the points for the monomials."""
    return [sp.Rational(i, k) for i in range(k + 1)]


def setup_dofs(k: int) -> list[sp.Symbol]:
    """Setup the degrees of freedom for the monomials."""
    return list(sp.symbols(f"a:{k}"))


def setup_base_polynomial(basis: list[sp.Expr], dofs: list[sp.Symbol]) -> sp.Expr:
    """Setup the base polynomial for the monomials."""
    return sum(a * p for a, p in zip(dofs, basis, strict=True))


class Hermite:
    """Class for the Hermite polynomials."""

    def __init__(self, k: int):
        """Initialize the Hermite polynomials.

        Args:
            k (int): The number of points.
        """
        self.k = k
        self.n = 2 * (k + 1)
        self.basis = setup_basis(self.n)
        self.dofs = setup_dofs(self.n)
        self.points = setup_points(k)
        self.pol = setup_base_polynomial(self.basis, self.dofs)

    def solve(self) -> list[sp.Expr]:
        """Solve the Hermite polynomials.

        Returns:
            list[sp.Expr]: The Hermite polynomials.
        """
        hermite_basis = []

        for j in range(self.k + 1):
            eqs = []
            for i in range(self.n):
                x_i = self.points[i // 2]
                print(f"{x_i=}, {i % 2 = }")

                if i % 2 == 0:
                    pol = self.pol.subs(x, x_i)
                else:
                    pol = sp.diff(self.pol, x).subs(x, x_i)

                eq = pol - dirac(i // 2, j)
                eqs.append(eq)

            coeffs = sp.solve(eqs, self.dofs)
            H_j = self.pol.subs(coeffs)

            hermite_basis.append(H_j)

        return hermite_basis


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Hermite Polynomials")
    axs: list[plt.Axes]

    k = 4
    hermite = Hermite(k)
    hermite_polynomials = hermite.solve()

    x_vals = np.linspace(0, 1, 100)
    for i, H in enumerate(hermite_polynomials):
        H_deriv = sp.diff(H, x)
        H_func = sp.lambdify(x, H, modules=["numpy"])
        H_deriv_func = sp.lambdify(x, H_deriv, modules=["numpy"])
        axs[0].plot(x_vals, H_func(x_vals), label=f"$H_{i}$")
        axs[1].plot(x_vals, H_deriv_func(x_vals), label=f"$H_{i}'$")

    titles = ["Basis functions", "Derivatives"]
    y_labels = ["$H(x)$", "$H'(x)$"]
    x_ticks = np.linspace(0, 1, k + 1)
    x_ticks = hermite.points

    for ax, title, y_label in zip(axs, titles, y_labels, strict=True):
        ax.set_title(title)
        ax.set_ylabel(y_label)

    for ax in axs:
        ax.set_xlabel("$x$")
        ax.set_xticks([float(x) for x in x_ticks])
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()
