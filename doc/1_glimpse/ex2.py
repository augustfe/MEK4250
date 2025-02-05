import numpy as np
import sympy as sp
from tqdm import tqdm

x = sp.symbols("x", real=True)


def bernstein_polynomial(n: int, i: int) -> sp.Expr:
    return sp.binomial(n, i) * x**i * (1 - x) ** (n - i)


def form_basis(N: int) -> list[sp.Expr]:
    # Form all basis functions except the first and the last one
    # to satisfy Dirichlet boundary conditions
    return [bernstein_polynomial(N, i) for i in range(1, N)]


def test_function(coeffs: list[sp.Expr], basis: list[sp.Expr]) -> sp.Expr:
    return sum([coeff * basis_func for coeff, basis_func in zip(coeffs, basis)])


def assemble_A(N: int) -> sp.Matrix:
    basis = form_basis(N)
    print(len(basis))
    n = len(basis)
    A = sp.zeros(n, n)
    for i, b_i in tqdm(enumerate(basis)):
        N_i = b_i.diff(x)
        for j, b_j in enumerate(basis):
            N_j = b_j.diff(x)
            A[i, j] = sp.integrate(N_i * N_j, (x, 0, 1))

    return A


def assemble_b(N: int, f: sp.Expr) -> sp.Matrix:
    basis = form_basis(N)
    b = sp.zeros(len(basis), 1)
    for i, b_i in tqdm(enumerate(basis)):
        b[i] = sp.integrate(b_i * f, (x, 0, 1))

    return b


def solve_for_coeffs(N: int, f: sp.Expr = x**2) -> list[sp.Expr]:
    A = assemble_A(N)
    b = assemble_b(N, f)
    coeffs = A.solve(b)
    return coeffs


def true_u(x: float) -> float:
    return -(x**4) / 12 + x / 12


def compute_L2_error(N: int) -> float:
    coeffs = solve_for_coeffs(N)
    basis = form_basis(N)
    u_N = test_function(coeffs, basis)
    error = sp.sqrt(sp.integrate((u_N - true_u(x)) ** 2, (x, 0, 1)))
    return error


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 10
    coeffs = solve_for_coeffs(N)
    basis = form_basis(N)
    u_N = test_function(coeffs, basis)
    u_N = sp.lambdify(x, u_N, modules="numpy")
    x_vals = np.linspace(0, 1, 1000)

    plt.plot(x_vals, true_u(x_vals), label="True $u$")
    plt.plot(x_vals, u_N(x_vals), label=f"$u_{{{N}}}$")
    plt.legend()
    plt.show()

    Ns = list(range(2, 11))
    errors = [compute_L2_error(N) for N in Ns]
    plt.loglog(Ns, errors, marker="x")
    plt.xlabel("$N$")
    plt.ylabel("Error")
    plt.title("Convergence of the error")

    plt.show()

    errors = [sp.simplify(error) for error in errors]
    print(errors)
