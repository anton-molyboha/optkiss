import numpy as np
import scipy.optimize as sopt


class GradientDescentObjective(object):
    """
    An abstract class representing a function to be optimized
    """

    def __init__(self):
        self.x = None

    def set_point(self, x):
        """
        Select the point at which the function's value and gradient will be queried
        :param x: the point of interest
        :return: None
        """
        self.x = x

    def value(self):
        raise NotImplementedError("Abstract method")

    def gradient(self):
        raise NotImplementedError("Abstract method")

    def max_step(self, d):
        """If the function has some form of an inner barrier, the maximal step that can be taken from
        point x in the direction d such that to stay within the domain of the function."""
        return np.inf


class GradientDescent(object):
    def __init__(self, objective, x0):
        self.objective = objective
        self.x = x0

    def get_point(self):
        return self.x

    def iteration(self, lambda1, lambda2, max_step, eps):
        self.objective.set_point(self.x)
        obj0 = self.objective.value()
        grad_vec = self.objective.gradient()
        max_grad_step = max_step * self.objective.max_step(grad_vec)

        # Bisect to find suitable step size
        grad_vec_sq = np.dot(grad_vec, grad_vec)
        def objective_for_step(step):
            self.objective.set_point(self.x - step * grad_vec)
            return self.objective.value()
        def is_good(step):
            if step == 0:
                return -1
            obj = objective_for_step(step)
            if obj < obj0 - lambda2 * step * grad_vec_sq:
                return -1
            elif obj > obj0 - lambda1 * step * grad_vec_sq:
                return 1
            else:
                return 0
        if np.isfinite(max_grad_step) and is_good(max_grad_step) <= 0:
            step = max_grad_step
        else:
            if max_grad_step == np.inf:
                max_grad_step = 1.0
                while is_good(max_grad_step) < 0:
                    max_grad_step *= 2
            step = max_grad_step * sopt.bisect(lambda k: is_good(k * max_grad_step), 0, 1)

        # Move to the new point
        self.x = self.x - step * grad_vec
        return step * np.sqrt(grad_vec_sq) > eps

    def minimize(self, stopping_eps, grad_lambda1=0.3, grad_lambda2=0.6, max_step=0.3, iter_count=10000):
        while self.iteration(grad_lambda1, grad_lambda2, max_step, stopping_eps):
            iter_count -= 1
            if iter_count <= 0:
                raise RuntimeError("Exceeded iteration limit")
        return self.x
