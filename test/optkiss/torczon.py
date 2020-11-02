import numpy as np
import optkiss.torczon as ot
from scipy.optimize import rosen


def norm_implicit(n):
    def objf(x, y):
        if y < 0:
            return y - np.sum(x**2)
        return y**2 - np.sum(x**2)

    iteration = [0]
    def callback(x, y):
        iteration[0] += 1
        if iteration[0] % 100 == 0:
            print("Iteration: " + str(iteration[0]))

    x = ot.torczon_implicit(objf, 5*np.ones(n), np.sqrt(n), callback=callback)
    #print(x)
    return np.linalg.norm(x) < 1e-3


def norm_explicit(n):
    def objf(x):
        return np.sum(x**2)

    iteration = [0]
    def callback(x, y):
        iteration[0] += 1
        if iteration[0] % 100 == 0:
            print("Iteration: " + str(iteration[0]))

    x = ot.torczon(objf, 5*np.ones(n), callback=callback)
    #print(x)
    return np.linalg.norm(x) < 1e-3


def rosenbrock(n):
    def objf(x, y):
        return y - rosen(x)

    iteration = [0]
    def callback(x, y):
        iteration[0] += 1
        if iteration[0] % 100 == 0:
            print("Iteration: " + str(iteration[0]))

    x = ot.torczon_implicit(objf, np.zeros(n), 0.0, callback=callback)
    print(x)
    return np.linalg.norm(x - np.ones(n)) < 1e-3

if __name__ == "__main__":
    print("Testing minimization of ||x|| in R^3 as an implicit function: " + str(norm_implicit(3)))
    print("Testing minimization of ||x|| in R^{300} as an implicit function: " + str(norm_implicit(300)))
    print("Testing minimization of ||x|| in R^{300} as an explicit function: " + str(norm_explicit(300)))
    print("Testing minimization of Rosenbrock in R^2: " + str(rosenbrock(2)))
