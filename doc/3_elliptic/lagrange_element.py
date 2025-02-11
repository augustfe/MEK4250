import matplotlib.pyplot as plt
import sympy as sp

x, y = sp.symbols("x y")


def compute_nodes(n: int) -> list[sp.Expr]:
    """Compute the nodes of the Lagrange element of degree n on the reference triangle.

    The reference triangle is defined by the vertices (0, 0), (1, 0), and (0, 1).

    Args:
        n (int): The degree of the Lagrange element.

    Returns:
        list[sympy.Expr]: The n choose 2 nodes of the Lagrange element.
    """
    nodes = []
    for i in range(n + 1):
        for j in range(i + 1):
            x_coord = sp.Rational(j, n)
            y_coord = 1 - sp.Rational(i, n)
            nodes.append((x_coord, y_coord))

    return sorted(nodes)


def plot_nodes(nodes: list[sp.Expr]) -> None:
    """Plot the nodes of the Lagrange element on the reference triangle.

    Args:
        nodes (list[sympy.Expr]): The nodes of the Lagrange element.
    """

    _, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)

    for x, y in nodes:
        ax.plot(x, y, "o", color="blue")

    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], "k-")

    plt.show()


def monomial_basis(n: int) -> list[sp.Expr]:
    """Compute the monomial basis of degree upto n.

    Args:
        n (int): The degree of the Lagrange element.

    Returns:
        list[sympy.Expr]: The monomial basis of the Lagrange element.
    """
    basis = []
    for i in range(n + 1):
        for j in range(i + 1):
            basis.append(x ** (i - j) * y**j)

    return basis


def lagrange_coeffs(n: int) -> sp.Matrix:
    """Compute the coefficients of the Lagrange basis of degree n.

    Args:
        n (int): The degree of the Lagrange element.

    Returns:
        sp.Matrix: The coefficients of the Lagrange basis.
    """
    nodes = compute_nodes(n)
    m_basis = monomial_basis(n)

    len_basis = len(m_basis)

    A = sp.zeros(len_basis, len_basis)
    # Don't need to use B, as it is the identity matrix
    # B = sp.eye(len_basis)

    for i, (x_i, y_i) in enumerate(nodes):
        for j, phi in enumerate(m_basis):
            A[i, j] = phi.subs({x: x_i, y: y_i})

    C = A.inv()  # @ B
    return C


def lagrange_basis(n: int) -> list[sp.Expr]:
    """Compute the Lagrange basis of degree n.

    Args:
        n (int): The degree of the Lagrange element.

    Returns:
        list[sympy.Expr]: The Lagrange basis.
    """
    coeffs = lagrange_coeffs(n)
    m_basis = monomial_basis(n)

    basis_vector = sp.Matrix(m_basis)
    basis = coeffs.T @ basis_vector

    test_basis = sp.Matrix(m_basis)
    print(test_basis)

    return basis


if __name__ == "__main__":
    n = 2
    nodes = compute_nodes(n)
    # plot_nodes(nodes)
    print(nodes)
    sp.pprint(lagrange_basis(2))
