import numpy as np
import optkiss.gradient_descent as ot
from scipy.optimize import rosen, rosen_der


def norm_explicit(n):
    class Objf(ot.GradientDescentObjective):
        def __init__(self):
            super(Objf, self).__init__()

        def value(self):
            return np.sum(self.x ** 2)

        def gradient(self):
            return 2 * self.x

    objf = Objf()

    # iteration = [0]
    # def callback(x, y):
    #     iteration[0] += 1
    #     if iteration[0] % 100 == 0:
    #         print("Iteration: " + str(iteration[0]))

    gd = ot.GradientDescent(objf, 5 * np.ones(n))
    gd.minimize(1e-4)
    x = gd.x
    #print(x)
    return np.linalg.norm(x) < 1e-3


def rosenbrock(n):
    class Objf(ot.GradientDescentObjective):
        def __init__(self):
            super(Objf, self).__init__()

        def value(self):
            return rosen(self.x)

        def gradient(self):
            return rosen_der(self.x)

    objf = Objf()
    gd = ot.GradientDescent(objf, 5 * np.ones(n))
    gd.minimize(1e-4, iter_count=100000)
    x = gd.x
    print(x)
    return np.linalg.norm(x - np.ones(n)) < 1e-3


if __name__ == "__main__":
    print("Testing minimization of ||x|| in R^3 as an explicit function: " + str(norm_explicit(3)))
    print("Testing minimization of ||x|| in R^{300} as an explicit function: " + str(norm_explicit(300)))
    print("Testing minimization of Rosenbrock in R^2: " + str(rosenbrock(2)))
    # print("Testing minimization of Rosenbrock in R^4: " + str(rosenbrock(4)))
