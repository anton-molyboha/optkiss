import numpy as np
import optkiss.torczon as ot

def norm(n):
    def objf(x, y):
        if y < 0:
            y = 0.0
        return y**2 - np.sum(x**2)

    x = ot.torczon_implicit(objf, 5*np.ones(n), np.sqrt(n))
    #print(x)
    return np.linalg.norm(x) < 1e-3

if __name__ == "__main__":
    print("Testing minimization of ||x|| in R^3: " + str(norm(3)))
    print("Testing minimization of ||x|| in R^{300}: " + str(norm(300)))
