import numpy as np
from ex3 import main


def sine(x: float, k: int) -> float:
    return np.sin(k * np.pi * x[0])


if __name__ == "__main__":
    main(lambda x: sine(x, 10))
