import numpy as np


def torczon_implicit(f, x0, y0, callback=lambda x, y: None, initial_step_size=1.0):
    return TorczonImpl(callback=callback).torczon_implicit(f, x0, y0, initial_step_size=initial_step_size)


def torczon(f, x0, callback=lambda x, y: None, initial_step_size=1.0):
    return TorczonImpl(callback=callback).torczon(f, x0, initial_step_size=initial_step_size)


def _np_false(shape):
    return np.zeros(shape, np.bool)


def _np_true(shape):
    return np.ones(shape, np.bool)


def _update_range_implicit(f, x, bnds, y):
    c = f(x, y)
    if c < 0:
        return (y, bnds[1])
    elif c > 0:
        return (bnds[0], y)
    elif c == 0:
        return (y, y)
    else:  # c is NaN
        raise ValueError('Implicit function returned a NaN')


def _update_range_explicit(f, x, bnds, y):
    res = f(x)
    return (res, res)


class TorczonImpl:
    def __init__(self, callback=lambda x, y: None):
        self._torczon_implicit_function_calls = 0
        self._callback = callback
        self.expand_coef = 2.0
        self.contract_coef = 1 / self.expand_coef
        self.xeps = 1e-4
        self.yeps = 1e-4

    # torczon's derivative-free optimization algorithm
    # f_orig must be increasing in y
    def torczon_implicit(self, f, x0, y0, initial_step_size=1.0):
        def update_range(x, bnds, y):
            self._torczon_implicit_function_calls += 1
            return _update_range_implicit(f, x, bnds, y)
        return self.run(update_range, x0, y0, initial_step_size=initial_step_size)

    def torczon(self, f, x0, initial_step_size=1.0):
        def update_range(x, bnds, y):
            self._torczon_implicit_function_calls += 1
            return _update_range_explicit(f, x, bnds, y)
        return self.run(update_range, x0, f(x0), initial_step_size=initial_step_size)

    def run(self, update_range, x0, y0, initial_step_size=1.0):
        self._torczon_implicit_function_calls = 0
        callback = self._callback

        x0 = np.asarray(x0)
        n = len(x0)
        xsimplex = x0.reshape([n, 1]) * np.ones([1, n+1])
        xsimplex[:n, :n] = xsimplex[:n, :n] + initial_step_size * np.eye(n)

        #print("torczon_implicit: initializing")
        fvalsx = y0 + np.concatenate([np.zeros([1, n+1]), np.ones([1, n+1])], 0)
        fvalsx, bestxind = self._find_smallest(update_range, xsimplex, fvalsx, _np_false([n+1]))

        goon = True
        iteration_count = 1
        while goon:
            #print("torczon_implicit: iteration %d" % iteration_count)
            fvalsx, xsimplex, bestxind, finishx = self._torczon_step(fvalsx, xsimplex, bestxind, update_range)
            callback(xsimplex[:, bestxind], 0.5 * (fvalsx[0, bestxind] + fvalsx[1, bestxind]))

            # Check stopping condition
            if finishx:
                goon = False
            else:
                iteration_count += 1
            #print("torczon_implicit: function calls made: %d" % self._torczon_implicit_function_calls)

        x = xsimplex[:, bestxind]
        #y = (fvalsx[0,bestxind] + fvalsx[1,bestxind]) / 2
        return x #, y

    #function [fvals bestx] = find_smallest(f, x, fvals, initflag)
    def _find_smallest(self, update_range, x, fvals, initflag):
        # Dimension of the space
        n = x.shape[0]
        # Number of points
        m = x.shape[1]
        # Initialize all fvals
        #print("torczon_implicit.find_smallest: initializing fvals")
        for j in range(m):
            if not initflag[j]:
                step = fvals[1, j] - fvals[0, j]
                fvals[:, j] = update_range(x[:, j], (-np.inf, np.inf), fvals[0, j])
                while not np.isfinite(fvals[0, j]):
                    fvals[:, j] = update_range(x[:, j], fvals[:, j], fvals[1, j] - step)
                    step *= 2
                while not np.isfinite(fvals[1, j]):
                    fvals[:, j] = update_range(x[:, j], fvals[:, j], fvals[0, j] + step)
                    step *= 2
        # Break ties
        #print("torczon_implicit.find_smallest: breaking ties")
        goon = True
        while goon:
            bestx = np.argmin(fvals[0, :])
            goon = False
            improved = False
            for j in range(m):
                if (j != bestx) and (fvals[0, j] < fvals[1, bestx]):
                    goon = True
                    if fvals[1, j] - fvals[0, j] > self.yeps:
                        testy = (fvals[0, j] + fvals[1, j]) / 2
                        self._torczon_implicit_function_calls += 1
                        fvals[:, j] = update_range(x[:, j], fvals[:, j], testy)
                        improved = True
            if goon and (fvals[1, bestx] - fvals[0, bestx] > self.yeps):
                testy = (fvals[0, bestx] + fvals[1, bestx]) / 2
                self._torczon_implicit_function_calls += 1
                fvals[:, bestx] = update_range(x[:, bestx], fvals[:, bestx], testy)
                improved = True
            goon = improved
        # The output variables already have the correct values
        return fvals, bestx

    #def [fvals simplex bestind finish] = torczon_step(fvals, simplex, bestind, f)
    def _torczon_step(self, fvals, simplex, bestind, update_range):
        n = simplex.shape[0]
        simplex_refl = np.zeros([n, n+1])
        fvals_refl = np.zeros([2, n+1])
        simplex_cur = np.zeros([n, n+1])
        fvals_cur = np.zeros([2, n+1])

        goon = True
        finish = False
        while goon:
            # Reflect
            for i in range(n+1):
                if i == bestind:
                    simplex_refl[:, i] = simplex[:, i]
                    fvals_refl[:, i] = fvals[:, i]
                else:
                    simplex_refl[:, i] = 2 * simplex[:, bestind] - simplex[:, i]
                    fvals_refl[:, i] = fvals[:, bestind]
            initflag = _np_false([n+1])
            initflag[bestind] = True
            fvals_refl, bestind_refl = self._find_smallest(update_range, simplex_refl, fvals_refl, initflag)
            if bestind_refl != bestind:
                # expand
                for i in range(n+1):
                    if i == bestind:
                        simplex_cur[:, i] = simplex_refl[:, i]
                        fvals_cur[:, i] = fvals_refl[:, i]
                    else:
                        simplex_cur[:, i] = self.expand_coef * simplex_refl[:, i] +\
                                            (1 - self.expand_coef) * simplex_refl[:, bestind]
                        fvals_cur[:, i] = fvals_refl[:, i]
                subindex = _np_true([n+1])
                subindex[bestind] = False
                simplex_ex = np.concatenate([simplex_refl, simplex_cur[:, subindex]], 1)
                fvals_ex = np.concatenate([fvals_refl, fvals_cur[:, subindex]], 1)
                initflag = _np_false([2*n+1])
                initflag[:(n+1)] = _np_true([n+1])
                fvals_ex, bestind_ex = self._find_smallest(update_range, simplex_ex, fvals_ex, initflag)
                if bestind_ex >= n+1:
                    simplex = simplex_cur
                    fvals[:, subindex] = fvals_ex[:, (n+1):(2*n+1)]
                    fvals[:, bestind] = fvals_ex[:, bestind]
                    bestind_ex -= (n + 1)
                    if bestind_ex >= bestind:
                        bestind = bestind_ex + 1
                    else:
                        bestind = bestind_ex
                else:
                    simplex = simplex_refl
                    fvals = fvals_ex[:, :(n+1)]
                    bestind = bestind_ex
                goon = False
            else:
                # contract
                for i in range(n+1):
                    if i == bestind:
                        simplex_cur[:, i] = simplex[:, i]
                        fvals_cur[:, i] = fvals[:, i]
                    else:
                        simplex_cur[:, i] = self.contract_coef * simplex[:, i] +\
                                            (1 - self.contract_coef) * simplex[:, bestind]
                        fvals_cur[:, i] = fvals[:, i]
                initflag = _np_false([n+1])
                initflag[bestind] = True
                fvals_cur, bestind_cur = self._find_smallest(update_range, simplex_cur, fvals_cur, initflag)
                if bestind_cur != bestind:
                    goon = False
                    bestind = bestind_cur
                simplex = simplex_cur
                fvals = fvals_cur
                # check stopping condition
                finish = True
                for i in range(n+1):
                    if np.linalg.norm(simplex[:, i] - simplex[:, bestind]) > self.xeps:
                        finish = False
                if finish:
                    goon = False
        return fvals, simplex, bestind, finish
