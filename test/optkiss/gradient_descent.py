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
    stopping_eps = 1e-6
    iterations = [0]
    def stopping_condition(oldx, newx, grad_vec):
        iterations[0] += 1
        return objf.progress_metric(oldx, newx) < stopping_eps
    gd.minimize(stopping_eps, iter_count=100000, stopping_condition=stopping_condition)
    x = gd.x
    print("Result: {} in {} iterations.".format(x, iterations[0]))
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


def rosenbrock_BFGS(n):
    class Objf(ot.GradientDescentObjective):
        def __init__(self):
            super(Objf, self).__init__()

        def value(self):
            return rosen(self.x)

        def gradient(self):
            return rosen_der(self.x)

    objf = ot.BFGS(Objf())
    gd = ot.GradientDescent(objf, 5 * np.ones(n))
    stopping_eps = 1e-6
    iterations = [0]
    def stopping_condition(oldx, newx, grad_vec):
        iterations[0] += 1
        return objf.progress_metric(oldx, newx) < stopping_eps
    gd.minimize(1e-6, iter_count=100000, line_search="wolfe", stopping_condition=stopping_condition)
    x = gd.x
    print("Result: {} in {} iterations.".format(x, iterations[0]))
    return np.linalg.norm(x - np.ones(n)) < 1e-3


def normconstr_BFGS_inf(n):
    x0 = np.arange(2, n + 2)
    threshold = 1.0
    penalty_scale = 1e-6
    penalty_pwr = 1
    hit_inf = [0]
    class Objf(ot.GradientDescentObjective):
        def __init__(self):
            super(Objf, self).__init__()

        def value(self):
            if self.x[0] <= threshold:
                v = np.inf
                hit_inf[0] += 1
            else:
                v = np.dot(self.x, self.x) + np.sum(self.x) ** 2
                v += penalty_scale / (self.x[0] - threshold) ** penalty_pwr
            return v

        def gradient(self):
            v = rosen(self.x)
            if self.x[0] <= threshold:
                g = np.ones(len(self.x)) * np.nan
            else:
                g = 2 * self.x + 2 * np.sum(self.x) * np.ones(len(self.x))
                g[0] += -penalty_scale * penalty_pwr / \
                        (self.x[0] - threshold) ** (penalty_pwr + 1)
            return g

    objf = ot.BFGS(Objf())
    gd = ot.GradientDescent(objf, x0)
    stopping_eps = 1e-6
    iterations = [0]
    def stopping_condition(oldx, newx, grad_vec):
        iterations[0] += 1
        return objf.progress_metric(oldx, newx) < stopping_eps
    gd.minimize(1e-6, iter_count=100000, line_search="wolfe", stopping_condition=stopping_condition)
    x = gd.x
    expected = -np.ones(n) * threshold / n
    expected[0] = threshold
    print("Result: {} in {} iterations, hitting inf {} times.".format(x, iterations[0], hit_inf[0]))
    return np.linalg.norm(x - expected) < 1e-3


def scaled_gradient(n):
    class IterCounter(object):
        def __init__(self, eps):
            self.iters = 0
            self.eps = eps ** 2

        def __call__(self, oldx, newx, grad):
            self.iters += 1
            return np.sum((newx - oldx) ** 2) <= self.eps

    class Objf(ot.GradientDescentObjective):
        def __init__(self, scale):
            super(Objf, self).__init__()
            self.scale = scale

        def value(self):
            return np.sum((self.x / self.scale) ** 2)

        def gradient(self):
            return 2 * self.x / (self.scale ** 2)

    scale = 1 + np.arange(n) * 3
    objf = Objf(scale)
    x0 = 5 * np.ones(n)
    eps = 1e-5

    # iteration = [0]
    # def callback(x, y):
    #     iteration[0] += 1
    #     if iteration[0] % 100 == 0:
    #         print("Iteration: " + str(iteration[0]))

    gd0 = ot.GradientDescent(objf, x0)
    iter_counter0 = IterCounter(eps)
    gd0.minimize(eps, stopping_condition=iter_counter0)

    gd1 = ot.GradientDescent(ot.ScaledGradient(objf, scale), x0)
    iter_counter1 = IterCounter(eps)
    gd1.minimize(eps, iter_count=100, stopping_condition=iter_counter1)

    if not np.linalg.norm(gd0.x) <= 1e-3:
        print("Without scaled gradient: descent did not converge, ||x|| = {}".format(np.linalg.norm(gd0.x)))
        return False
    if not np.linalg.norm(gd1.x) <= 1e-3:
        print("With scaled gradient: descent did not converge, ||x|| = {}".format(np.linalg.norm(gd1.x)))
        return False
    print("Without scaled gradient: {} iterations, with scaled gradient: {} iterations".format(iter_counter0.iters, iter_counter1.iters))
    return iter_counter1.iters <= iter_counter0.iters



if __name__ == "__main__":
    print("Testing minimization of ||x|| in R^3 as an explicit function: " + str(norm_explicit(3)))
    print("Testing minimization of ||x|| in R^{300} as an explicit function: " + str(norm_explicit(300)))
    print("Testing minimization of Rosenbrock in R^2: " + str(rosenbrock(2)))
    # print("Testing minimization of Rosenbrock in R^4: " + str(rosenbrock(4)))
    print("Testing minimization of scaled Rosenbrock in R^2: " + str(rosenbrock_scaled(2)))
    print("Testing minimization of hierarchical Rosenbrock in R^2: " + str(rosenbrock_hierarchical(2)))
    print("Testing minimization of Rosenbrock in R^2 using BFGS method: " + str(rosenbrock_BFGS(2)))
    print("Testing minimization with inner penalty in R^2 using BFGS method: " + str(normconstr_BFGS_inf(2)))
    print("Testing ScaledGradient: " + str(scaled_gradient(3)))
