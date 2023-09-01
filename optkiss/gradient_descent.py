import math
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

    def descent_direction(self):
        """
        The direction in which to look for the next iteration.

        Must have: np.dot(descent_direction(), gradient()) < 0

        By default, this is just -gradient()

        A perfect value would be -np.solve(hermitian, gradient()), but that would require
        to compute the matrix of second derivatives (aka hermitian)
        :return:
        """
        return -self.gradient()

    def progress_metric(self, x1, x2):
        """
        How much of a progress is it to move from x1 to x2? Used in the stopping criterion.

        By default, just the Euclidean distance from x1 to x2.
        :param x1:
        :param x2:
        :return:
        """
        return np.linalg.norm(x2 - x1)

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

    def iteration(self, lambda1, lambda2, max_step, eps, stopping_condition=None, line_search="armijo"):
        self.objective.set_point(self.x)
        obj0 = self.objective.value()
        grad_vec = self.objective.gradient()
        direction = self.objective.descent_direction()
        max_grad_step = max_step * self.objective.max_step(direction)

        # Bisect to find suitable step size
        grad_vec_sq = -np.dot(grad_vec, direction)
        grad_vec_len = np.sqrt(grad_vec_sq)
        if grad_vec_sq * 1e20 < eps:
            # Within computational accuracy, gradient is zero. Stop iterations.
            return False
        if line_search == "wolfe":
            def objective_for_step(step):
                self.objective.set_point(self.x + step * direction)
                v = self.objective.value()
                if np.isfinite(v):
                    g = self.objective.gradient()
                else:
                    g = np.nan * np.ones(len(self.x))
                return v, g
            def is_good(step):
                obj, g = objective_for_step(step)
                if obj == np.inf:
                    return 1
                if obj > obj0 - lambda1 * step * grad_vec_sq:
                    return 1
                elif -np.dot(g, direction) > lambda2 * grad_vec_sq:
                    return -1
                else:
                    return 0
        elif line_search == "armijo":
            # Default is the Armijo rule for upper bound of the step, and a similar
            # inequality for the lower bound.
            def objective_for_step(step):
                self.objective.set_point(self.x + step * direction)
                return self.objective.value()
            def is_good(step):
                if step * grad_vec_len <= 0.5 * eps:
                    return -1
                obj = objective_for_step(step)
                if obj < obj0 - lambda2 * step * grad_vec_sq:
                    return -1
                elif obj > obj0 - lambda1 * step * grad_vec_sq:
                    return 1
                else:
                    return 0
        else:
            raise ValueError("Unknown line_search method: {}".format(line_search))
        if np.isfinite(max_grad_step) and is_good(max_grad_step) <= 0:
            step = max_grad_step
        else:
            if max_grad_step == np.inf:
                max_grad_step = 1.0
                while is_good(max_grad_step) < 0:
                    max_grad_step *= 2
            step_k = sopt.bisect(lambda k: is_good(k * max_grad_step), 0, 1, xtol=1e-20)
            if is_good(step_k * max_grad_step) > 0:
                # When is_good == 1, the objective function might be infinite, and we don't want that.
                step_k = max(0, min(step_k - 1e-20, math.nextafter(step_k, 0.0)))
            step = max_grad_step * step_k

        # Move to the new point
        newx = self.x + step * direction
        if stopping_condition is None:
            res = self.objective.progress_metric(self.x, newx) > eps
        else:
            res = not stopping_condition(self.x, newx, grad_vec)
        self.x = newx
        return res

    def minimize(self, stopping_eps, grad_lambda1=0.3, grad_lambda2=0.6, max_step=0.3, iter_count=10000,
                 stopping_condition=None, line_search="armijo"):
        while self.iteration(grad_lambda1, grad_lambda2, max_step, stopping_eps, stopping_condition, line_search):
            iter_count -= 1
            if iter_count <= 0:
                raise RuntimeError("Exceeded iteration limit")
        return self.x


class ScaledObjective(GradientDescentObjective):
    def __init__(self, base, scale):
        super(ScaledObjective, self).__init__()
        self.base = base
        self.scale = np.asarray(scale)

    def unscale(self, x):
        """Convert coordinates from the scaled space to the original space"""
        return x / self.scale

    def scale(self, x):
        """Convert coordinates from the original space to the scaled space"""
        return x * self.scale

    def set_point(self, x):
        self.base.set_point(self.unscale(x))

    def value(self):
        return self.base.value()

    def gradient(self):
        return self.base.gradient() / self.scale

    def max_step(self, d):
        return self.base.max_step(d / self.scale)


class ScaledGradient(GradientDescentObjective):
    """
    Modify descent direction as if we have scaled each coordinate of the decision variables space by a given factor.

    Unlike `ScaledObjective`, this class operates in the original space, only adjusts the descent direction.

    We are given a base objective function, f(x), and a scaling factor, `scale`.
    Consider a "scaled" decision variable space: y = x / `scale` (the division is coordinate-wise).
    A function g(y) is such that g(y) = g(x / `scale`) = f(x).
    Our intent is to apply gradient descent to g(y).
    g(y) = f(y * `scale`) , so gradient of g(y) is dg/dy = df/dx * `scale`.
    The descent direction for the gradient descent method in the y-space would be -dg/dy = -df/dx * `scale`.
    However, in the x-space this corresponds to the same multiplied by `scale` once again: -df/dx * (`scale` ** 2).
    This is our modified descent direction.
    """
    def __init__(self, base, scale):
        """
         `base`: the objective function to modify, f(x).
         `scale':
        """
        super(ScaledGradient, self).__init__()
        self.base = base
        self.scale = np.asarray(scale)

    def set_point(self, x):
        self.base.set_point(x)

    def value(self):
        return self.base.value()

    def gradient(self):
        return self.base.gradient()

    def descent_direction(self):
        return -self.base.gradient() * (self.scale ** 2)

    def max_step(self, d):
        return self.base.max_step(d)

    def progress_metric(self, x1, x2):
        return self.base.progress_metric(x1, x2)


class HierarchicalElement(GradientDescentObjective):
    def __init__(self, base, combined_x, stage_inds, minimization_params, lower_element=None):
        """
        An element of a hierarchical optimization process
        :param base:
        :param stages: ordered from inner to outer
        :param index:
        :param lower_element:
        """
        super(HierarchicalElement, self).__init__()
        self.base = base
        self.combined_x = combined_x
        self.stage_inds = stage_inds
        self.minimization_params = minimization_params
        self.lower_element = lower_element

    def set_point(self, x):
        self.combined_x[self.stage_inds] = x
        if self.lower_element is None:
            self.base.set_point(self.combined_x)
        else:
            gd = GradientDescent(self.lower_element, self.lower_element.get_point())
            gd.minimize(**self.minimization_params)
            self.lower_element.set_point(gd.x)

    def get_point(self):
        return np.array(self.combined_x[self.stage_inds])

    def get_combined_point(self):
        return np.array(self.combined_x)

    def value(self):
        return self.base.value()

    def gradient(self):
        return self.base.gradient()[self.stage_inds]

    def max_step(self, d):
        full_d = np.zeros(len(self.combined_x))
        full_d[self.stage_inds] = d
        return self.base.max_step(full_d)


def hierarchical_objective(base, stages, x0, minimization_params):
    res = None
    combined_x = np.empty(len(x0))
    combined_x[:] = x0
    for stage in stages:
        res = HierarchicalElement(base, combined_x, stage, minimization_params, res)
    return res


"""
An adapter of a GradientDescentObjective that implements BFGS method.
BFGS method is one of quasi-Newton methods, and has super-linear convergence
under suitable conditions.
"""
class BFGS(GradientDescentObjective):
    def __init__(self, base):
        super(BFGS, self).__init__()
        self._base = base
        self._H = None
        self._last_x = None
        self._last_gradient = None

    def set_point(self, x):
        if self._H is None:
            self._H = np.eye(len(x))
        self.x = x
        self._base.set_point(x)

    def value(self):
        return self._base.value()

    def gradient(self):
        return self._base.gradient()

    def descent_direction(self):
        grad = self._base.gradient()
        if self._H is None:
            self._H = np.eye(len(grad))
        if ((self._last_x is not None) and
            (self._last_gradient is not None)):
            step = self.x - self._last_x
            delta_grad = grad - self._last_gradient
            sg = np.dot(step, delta_grad)
            # The sufficient conditions for having positive-definite H.
            # We don't want to update H otherwise, just in case.
            if sg > 0:
                r = np.dot(self._H, delta_grad) / sg
                sr = np.dot(step[:, np.newaxis], r[np.newaxis, :])
                self._H += ((1 + np.dot(r, delta_grad)) / sg *
                                np.dot(step[:, np.newaxis], step[np.newaxis, :]) -
                            (sr + sr.T))
        res = -np.dot(self._H, grad)
        self._last_x = self.x
        self._last_gradient = grad
        return res

    def progress_metric(self, x1, x2):
        return self._base.progress_metric(x1, x2)

    def max_step(self, d):
        return self._base.max_step(d)


"""
An adapter of a GradientDescentObjective that makes sure that the gradient is
finite whenever the objective value is finite.
"""
class AssertFinite(GradientDescentObjective):
    def __init__(self, base: GradientDescentObjective):
        super().__init__()
        self._base = base
        self.x = base.x

    def set_point(self, x):
        self._base.set_point(x)
        super().set_point(x)

    def value(self):
        return self._base.value()

    def gradient(self):
        v = self._base.value()
        if not np.isfinite(v):
            raise RuntimeError("Objective value is non-finite, gradient should not have been called. Point: {}".format(self.x))
        res = self._base.gradient()
        if not np.all(np.isfinite(res)):
            raise RuntimeError("Non-finite gradient despite finite objective value. Point: {}".format(self.x))
        return res

    def descent_direction(self):
        v = self._base.value()
        if not np.isfinite(v):
            raise RuntimeError("Objective value is non-finite, descent_direction should not have been called. Point: {}".format(self.x))
        res = self._base.descent_direction()
        if not np.all(np.isfinite(res)):
            raise RuntimeError("Non-finite descent direction despite finite objective value. Point: {}".format(self.x))
        return res

    def progress_metric(self, x1, x2):
        return self._base.progress_metric(x1, x2)

    def max_step(self, d):
        return self._base.max_step(d)