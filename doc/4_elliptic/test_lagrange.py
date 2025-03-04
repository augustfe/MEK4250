import pytest
import sympy as sp
from lagrange_element import compute_nodes, lagrange_basis

x, y = sp.symbols("x y")


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_lagrange(n) -> None:
    basis = lagrange_basis(n)
    nodes = compute_nodes(n)

    for i, (x_i, y_i) in enumerate(nodes):
        assert basis[i].subs({x: x_i, y: y_i}) == 1
        for j, (x_j, y_j) in enumerate(nodes):
            if i != j:
                assert basis[i].subs({x: x_j, y: y_j}) == 0
