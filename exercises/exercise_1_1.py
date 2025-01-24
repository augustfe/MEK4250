import numpy as np


def compute_coeffs(N: int) -> np.ndarray:
    ell = np.arange(1, N + 1)

    diag = np.ones(N) * 1 / 2
    # diag[0] = 1
    diag *= (np.pi * ell) ** 2
    A = np.diag(diag)

    alternating = (-1) ** ell
    # b = alternating / (np.pi * ell) + 2 / (np.pi * ell) ** 3 * (alternating - 1)
    b = ((2 - np.pi**2 * ell**2) * alternating - 2) / (np.pi**3 * ell**3)

    # assert np.allclose(b, b1)

    return np.linalg.solve(A, b)


def true_u(x: np.ndarray) -> np.ndarray:
    return -1 / 12 * x**4 + 1 / 12 * x


def u_N(x: np.ndarray, N: int) -> np.ndarray:
    coeffs = compute_coeffs(N)
    ell = np.arange(1, N + 1)
    un = np.vectorize(lambda x: np.sum(coeffs * np.sin(ell * np.pi * x)))
    return un(x)


def compute_error(N: int, ord: str | int) -> float:
    x = np.linspace(0, 1, 1000)

    import matplotlib.pyplot as plt

    plt.plot(x, true_u(x), label="True solution")
    plt.plot(x, u_N(x, N), label=f"Approximation with N={N}")
    plt.legend()
    plt.show()

    return np.linalg.norm(true_u(x) - u_N(x, N), ord=ord)


if __name__ == "__main__":
    Ns = [10, 20, 40]
    for N in Ns:
        print(f"Error for N = {N}: {compute_error(N, ord=2)}")
        print(f"Error for N = {N}: {compute_error(N, ord=np.inf)}")
