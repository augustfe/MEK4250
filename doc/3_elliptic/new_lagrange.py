import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x, y = sp.symbols("x y")


def compute_reference_lines(d: int) -> list[sp.Expr]:
    """Compute the reference lines of the Lagrange element of degree d on the reference triangle.

    The reference triangle is defined by the vertices (0, 0), (1, 0), and (0, 1).

    Args:
        d (int): The degree of the Lagrange element.

    Returns:
        list[sympy.Expr]: The reference lines of the Lagrange element.
    """
    # lines = []
    horizontal_lines = []
    vertical_lines = []
    diagonal_lines = []
    for i in range(d + 1):
        dline = x + y - sp.Rational(i, d)
        vline = x - sp.Rational(i, d)
        hline = y - sp.Rational(i, d)

        diagonal_lines.append(dline)
        vertical_lines.append(vline)
        horizontal_lines.append(hline)

    return diagonal_lines, vertical_lines, horizontal_lines


def plot_reference_lines(lines: list[sp.Expr]) -> None:
    """Plot the reference lines of the Lagrange element on the reference triangle.

    Args:
        lines (list[sympy.Expr]): The reference lines of the Lagrange element.
    """

    P = sp.plot_implicit(
        lines[0], (x, -0.5, 1.5), (y, -0.5, 1.5), line_color="blue", show=False
    )

    for line in lines[1:]:
        P2 = sp.plot_implicit(
            line, (x, -0.5, 1.5), (y, -0.5, 1.5), line_color="blue", show=False
        )
        P.append(P2[0])

    P.show()


def compute_control_points(d: int) -> list[sp.Expr]:
    """Compute the control points of the Lagrange element of degree d on the reference triangle.

    The reference triangle is defined by the vertices (0, 0), (1, 0), and (0, 1).

    Args:
        d (int): The degree of the Lagrange element.

    Returns:
        list[sympy.Expr]: The control points of the Lagrange element.
    """
    # loop over i,j,k such that i+j+k = d
    # where I need to be able to index control_points[i][j][k]
    control_points = []
    for i in range(d + 1):
        control_points.append([])
        for j in range(d + 1 - i):
            control_points[i].append([])
            for k in range(d + 1 - i - j):
                control_points[i][j].append((sp.Rational(k, d), sp.Rational(j, d)))

    return control_points


def plot_lines_and_control_points(
    lines: list[sp.Expr],
    control_points: list[sp.Expr],
    indices: list[tuple[int, int, int]],
):
    """Plot the reference lines and control points of the Lagrange element on the reference triangle.

    Args:
        lines (list[sympy.Expr]): The reference lines of the Lagrange element.
        control_points (list[sympy.Expr]): The control points of the Lagrange element.
        indices (list[tuple[int, int, int]]): The indices of the control points to plot.
    """
    _, ax = plt.subplots()
    ax.set_aspect("equal")
    # ax.set_xlim(-0.5, 1.5)
    # ax.set_ylim(-0.5, 1.5)

    types = ["diagonal", "vertical", "horizontal"]

    d = len(control_points) - 1

    for type, type_line in zip(types, lines, strict=True):
        # if type in ("vertical", "horizontal"):
        #     continue
        for i, line in enumerate(type_line):
            f = sp.lambdify((x, y), line, modules="numpy")
            print(line)
            match type:
                case "diagonal":
                    xmin, xmax = 0, (i) / d
                    ymin, ymax = (i) / d, 0
                case "vertical":
                    xmin, xmax = (d - i) / d, (d - i) / d
                    ymin, ymax = 0, i / d
                case "horizontal":
                    xmin, xmax = 0, (i) / d
                    ymin, ymax = (d - i) / d, (d - i) / d

            x_vals = np.linspace(xmin, xmax, 100)
            y_vals = np.linspace(ymin, ymax, 100)
            # ax.plot(x_vals, f(x_vals, y_vals), color="blue")
            ax.plot([xmin, xmax], [ymin, ymax], color="blue")
            # ax.plot(x_vals, y_vals, f(x_vals, y_vals), color="blue")
            print(f"Plotted {type} line {i}")

    for i, j, k in indices:
        y_coord, x_coord = control_points[i][j][k]
        ax.plot(x_coord, y_coord, "o", color="red")
        text = r"$\xi_{" + f"{i}, {j}, {k}" + r"}$"
        ax.text(x_coord, y_coord, text, fontsize=12)

    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], "k-")

    plt.show()


if __name__ == "__main__":
    d = 4
    # diagonal_lines, vertical_lines, horizontal_lines = compute_reference_lines(d)
    # plot_reference_lines(diagonal_lines)
    # plot_reference_lines(vertical_lines)
    # plot_reference_lines(horizontal_lines)
    lines = compute_reference_lines(d)
    control_points = compute_control_points(d)
    print(control_points[4][0][0])
    print(control_points[3][1][0])
    print(control_points[3][0][1])

    print(control_points[0][0][4])
    print(control_points[0][4][0])

    indices = [(4, 0, 0), (3, 1, 0), (3, 0, 1), (0, 0, 4), (0, 4, 0)]

    plot_lines_and_control_points(lines, control_points, indices)
