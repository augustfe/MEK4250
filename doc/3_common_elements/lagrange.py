"""Script for defining the Lagrange polynomials on the unit triangle. Exercise 3.3"""

from typing import Literal

import sympy as sp

x, y = sp.symbols("x y")


class Lines:
    """Class for the lines on the unit triangle in one direction.
    
    The vertices of the trinangle are numbered as follows:
        2
        | \
        |  \
        |   \
        0 -- 1
    """

    def __init__(self, k: int, opposite_vertex: Literal[0, 1, 2]) -> None:
        """Initialize the lines.

        Args:
            k (int): The number of lines
            opposite_vertex (Literal[0, 1, 2]): The vertex opposite to the lines.
        """
        self.k = k
        self.opposite = opposite_vertex

    @property
    def intervals(self) -> list[sp.Rational]:
        """Get the intervals for the lines."""
        vals = [sp.Rational(i, self.k) for i in range(self.k + 1)]
        if self.opposite == 0:
            vals = vals[::-1]

        return vals

    @property
    def base_expr(self) -> sp.Expr:
        """Get the base expression for the lines."""
        match self.opposite:
            case 0:
                return x + y
            case 1:
                return x
            case 2:
                return y

    def __len__(self) -> int:
        """Get the number of lines."""
        return self.k + 1

    def __getitem__(self, i: int) -> sp.Expr:
        """Get the i-th line.

        Args:
            i (int): The index of the line.

        Returns:
            sp.Expr: The i-th line, counted towards the opposite vertex.
        """
        return self.base_expr - self.intervals[i]


class UnitTriangleLagrange:
    """Class for a Lagrange polynomials on the unit triangle."""

    def __init__(self, k: int) -> None:
        """Initialize the Lagrange polynomials.

        Args:
            k (int): The order of the polynomial.
        """
        self.k = k
        self.lines = [Lines(k, i) for i in range(3)]

    def get_pol(self, i: int, j: int) -> sp.Expr:
        """Get the polynomial at (i / k, j / k).

        Args:
            i (int): The index in the x direction.
            j (int): The index in the y direction.

        Returns:
            sp.Expr: The polynomial at (i / k, j / k).
        """
        if i + j > self.k:
            raise ValueError(
                "Polynomials are only defined within the triangle, "
                f"when i + j <= k. Here {i} + {j} > {self.k}."
            )
        x_coord = sp.Rational(i, self.k)
        y_coord = sp.Rational(j, self.k)

        a = self.k - i - j
        b = i
        c = j

        a_lines = [self.lines[0][i] for i in range(a)]
        b_lines = [self.lines[1][i] for i in range(b)]
        c_lines = [self.lines[2][i] for i in range(c)]

        all_lines = a_lines + b_lines + c_lines

        numerator: sp.Expr = sp.prod(all_lines)
        denominator = numerator.subs({x: x_coord, y: y_coord})

        return numerator / denominator


def test_lagrange(i_idx: int, j_idx: int, k: int = 4) -> None:
    """Check that the polynomial is 1 at (i / k, j / k) and 0 at all other points.

    Args:
        i (int): i-index
        j (int): j-index
        k (int, optional): The order of the polynomial. Defaults to 4.
    """
    lagrange = UnitTriangleLagrange(k)
    pol = lagrange.get_pol(i_idx, j_idx)

    for i in range(k + 1):
        for j in range(k + 1 - i):
            x_coord = sp.Rational(i, k)
            y_coord = sp.Rational(j, k)

            if i == i_idx and j == j_idx:
                assert pol.subs({x: x_coord, y: y_coord}) == 1
            else:
                assert pol.subs({x: x_coord, y: y_coord}) == 0


if __name__ == "__main__":
    k = 4
    lagrange = UnitTriangleLagrange(k)

    all_polynomials: list[list[sp.Expr]] = []
    for i in range(k + 1):
        tmp_pols = []
        for j in range(k + 1 - i):
            pol = lagrange.get_pol(i, j)
            tmp_pols.append(pol)
        all_polynomials.append(tmp_pols)

    for i_idx in range(k + 1):
        for j_idx in range(k + 1 - i_idx):
            num_ones = 0
            for i in range(k + 1):
                for j in range(k + 1 - i):
                    target = 0
                    if i == i_idx and j == j_idx:
                        target = 1
                    pol = all_polynomials[i_idx][j_idx]
                    x_coord = sp.Rational(i, k)
                    y_coord = sp.Rational(j, k)
                    val = pol.subs({x: x_coord, y: y_coord})
                    assert (
                        pol.subs({x: x_coord, y: y_coord}) == target
                    ), f"Failed for {i=}, {j=}, {i_idx=}, {j_idx=}"
