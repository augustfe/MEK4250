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

    return np.linalg.norm(true_u(x) - u_N(x, N), ord=ord)


def plot(N: int) -> None:
    from pathlib import Path

    import matplotlib.pyplot as plt

    savedir = Path(__file__).parents[1] / "figs"

    assert savedir.exists()

    plt.figure(figsize=(4, 3))
    x = np.linspace(0, 1, 250)
    plt.plot(x, true_u(x), label="True $u$")
    plt.plot(x, u_N(x, N), label=f"$u_{{{N}}}$")
    plt.legend()
    plt.title(f"Approximation of $u$ with $N = {N}$")
    plt.xlabel("$x$")

    plt.savefig(savedir / "exercise_1_1.pdf", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(4, 3))
    coeffs = compute_coeffs(N)
    plt.scatter(np.arange(1, N + 1), np.abs(coeffs), marker="x")
    plt.title(f"Coefficients for $N = {N}$")
    plt.xlabel("$n$")
    plt.ylabel("$u_n$")
    plt.yscale("log")
    plt.savefig(savedir / "exercise_1_1_coeffs.pdf", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    import pandas as pd

    Ns = [10, 20, 40]

    results = np.zeros((len(Ns), 2))
    for i, N in enumerate(Ns):
        results[i, 0] = compute_error(N, ord=2)
        results[i, 1] = compute_error(N, ord=np.inf)

    df = pd.DataFrame(results, index=Ns, columns=[r"$L_2$", r"$L_\infty$"])
    print(df.to_latex())

    plot(Ns[-1])
