import numpy as np
import optkiss.torczon as ot
from scipy.optimize import rosen


def norm(n):
    def objf(x, y):
        if y < 0:
            y = 0.0
        return y**2 - np.sum(x**2)

    iteration = [0]
    def callback(x, y):
        iteration[0] += 1
        if iteration[0] % 100 == 0:
            print("Iteration: " + str(iteration[0]))

    x = ot.torczon_implicit(objf, 5*np.ones(n), np.sqrt(n), callback=callback)
    #print(x)
    return np.linalg.norm(x) < 1e-3


def rosenbrock(n):
    def objf(x, y):
        return y - rosen(x)
    x = ot.torczon_implicit(objf, np.zeros(n), 0.0)
    print(x)
    return np.linalg.norm(x - np.ones(n)) < 1e-3

if __name__ == "__main__":
    print("Testing minimization of ||x|| in R^3: " + str(norm(3)))
    print("Testing minimization of ||x|| in R^{300}: " + str(norm(300)))
    print("Testing minimization of Rosenbrock in R^2: " + str(rosenbrock(2)))
