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


def rosenbrock_scaled(n):
    class Objf(ot.GradientDescentObjective):
        def __init__(self):
            super(Objf, self).__init__()

        def value(self):
            return rosen(self.x)

        def gradient(self):
            return rosen_der(self.x)

    objf = ot.ScaledObjective(Objf(), [10**i for i in range(n)])
    gd = ot.GradientDescent(objf, 5 * np.ones(n))
    gd.minimize(1e-4, iter_count=100000)
    x = objf.unscale(gd.x)
    print(x)
    return np.linalg.norm(x - np.ones(n)) < 1e-3


def rosenbrock_hierarchical(n):
    class Objf(ot.GradientDescentObjective):
        def __init__(self):
            super(Objf, self).__init__()

        def value(self):
            return rosen(self.x)

        def gradient(self):
            return rosen_der(self.x)

    objf = ot.hierarchical_objective(Objf(), [[i] for i in range(n - 1, -1, -1)], 5 * np.ones(n), {'stopping_eps': 1e-4, 'iter_count': 1000})
    gd = ot.GradientDescent(objf, objf.get_point())
    gd.minimize(1e-4, iter_count=1000)
    objf.set_point(gd.x)
    x = objf.get_combined_point()
    print(x)
    return np.linalg.norm(x - np.ones(n)) < 1e-3


if __name__ == "__main__":
    print("Testing minimization of ||x|| in R^3 as an explicit function: " + str(norm_explicit(3)))
    print("Testing minimization of ||x|| in R^{300} as an explicit function: " + str(norm_explicit(300)))
    print("Testing minimization of Rosenbrock in R^2: " + str(rosenbrock(2)))
    # print("Testing minimization of Rosenbrock in R^4: " + str(rosenbrock(4)))
    print("Testing minimization of scaled Rosenbrock in R^2: " + str(rosenbrock_scaled(2)))
    print("Testing minimization of hierarchical Rosenbrock in R^2: " + str(rosenbrock_hierarchical(2)))
