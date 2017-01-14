import numpy as np
import optkiss.torczon as ot

def norm(n):
    def objf(x, y):
        if y < 0:
            y = 0.0
        return y**2 - np.sum(x**2)

    x, y = ot.torczon_implicit(objf, 5*np.ones(n), np.sqrt(n))
    return np.linalg.norm(x) < 1e-3 and np.abs(y) < 1e-3

if __name__ == "__main__":
    print("Testing minimization of ||x||: " + str(norm(3)))
