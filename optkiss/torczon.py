import numpy as np


def _np_false(shape):
    return np.zeros(shape, np.bool)


def _np_true(shape):
    return np.ones(shape, np.bool)


def torczon_implicit(f, x0, y0):
    return TorczonImpl().torczon_implicit(f, x0, y0)


class TorczonImpl:
    def __init__(self):
        self._torczon_implicit_function_calls = 0
        self.expand_coef = 2.0
        self.contract_coef = 1 / self.expand_coef
        self.xeps = 1e-4
        self.yeps = 1e-4


    # torczon's derivative-free optimization algorithm
    # f_orig must be increasing in y
    def torczon_implicit(self,f,x0,y0):
        self._torczon_implicit_function_calls = 0

        x0 = np.asarray(x0)
        n = len(x0)
        xsimplex = x0.reshape([n, 1]) * np.ones([1, n+1])
        xsimplex[:n,:n] = xsimplex[:n,:n] + np.eye(n)

        print("torczon_implicit: initializing")
        fvalsx = np.concatenate([np.zeros([1,n+1]) , np.ones([1,n+1])], 0)
        fvalsx, bestxind = self._find_smallest(f, xsimplex, fvalsx, _np_false([n+1]))

        goon = True
        iteration_count = 1
        _torczon_implicit_function_calls = 0
        while goon:
            print("torczon_implicit: iteration %d" % iteration_count)
            fvalsx, xsimplex, bestxind, finishx = self._torczon_step(fvalsx,xsimplex,bestxind,f)

            # bestxind
            # xsimplex
            # fvalsx

            # Check stopping condition
            if finishx:
                goon = False
            else:
                iteration_count = iteration_count + 1
            print("torczon_implicit: function calls made: %d" % self._torczon_implicit_function_calls)

        x = xsimplex[:,bestxind]
        y = (fvalsx[0,bestxind] + fvalsx[1,bestxind]) / 2
        return x, y

    #function [fvals bestx] = find_smallest(f, x, fvals, initflag)
    def _find_smallest(self, f, x, fvals, initflag):
        # Dimension of the space
        n = x.shape[0]
        # Number of points
        m = x.shape[1]
        # Initialize all fvals
        print("torczon_implicit.find_smallest: initializing fvals")
        for j in range(m):
            if not initflag[j]:
                if f(x[:,j], fvals[0,j]) > 0:
                    step = fvals[1,j] - fvals[0,j]
                    goon = True
                    while goon:
                        fvals[1,j] = fvals[0,j]
                        fvals[0,j] = fvals[1,j] - step
                        step = 2 * step
                        self._torczon_implicit_function_calls = self._torczon_implicit_function_calls + 1
                        if f(x[:,j], fvals[0,j]) <= 0:
                            goon = False
                elif f(x[:,j], fvals[1,j]) < 0:
                    step = fvals[1,j] - fvals[0,j]
                    goon = True
                    while goon:
                        fvals[0,j] = fvals[1,j]
                        fvals[1,j] = fvals[0,j] + step
                        step = 2 * step
                        self._torczon_implicit_function_calls = self._torczon_implicit_function_calls + 1
                        if f(x[:,j], fvals[1,j]) >= 0:
                            goon = False
        # Break ties
        print("torczon_implicit.find_smallest: breaking ties")
        goon = True
        while goon:
            bestx = np.argmin(fvals[0,:])
            goon = False
            improved = False
            for j in range(m):
                if (j != bestx) and (fvals[0,j] < fvals[1,bestx]):
                    goon = True
                    if (fvals[1, j] - fvals[0, j] > self.yeps):
                        testy = (fvals[0,j] + fvals[1,j]) / 2
                        self._torczon_implicit_function_calls = self._torczon_implicit_function_calls + 1
                        if f(x[:,j], testy) <= 0:
                            fvals[0,j] = testy
                        else:
                            fvals[1,j] = testy
                        improved = True
            if goon and (fvals[1,bestx] - fvals[0,bestx] > self.yeps):
                testy = (fvals[0,bestx] + fvals[1,bestx]) / 2
                self._torczon_implicit_function_calls = self._torczon_implicit_function_calls + 1
                if f(x[:,bestx], testy) >= 0:
                    fvals[1,bestx] = testy
                else:
                    fvals[0,bestx] = testy
                improved = True
            goon = improved
        # The output variables already have the correct values
        return fvals, bestx

    #def [fvals simplex bestind finish] = torczon_step(fvals, simplex, bestind, f)
    def _torczon_step(self, fvals, simplex, bestind, f):
        n = simplex.shape[0]
        simplex_refl = np.zeros([n,n+1])
        fvals_refl = np.zeros([2,n+1])
        simplex_cur = np.zeros([n,n+1])
        fvals_cur = np.zeros([2,n+1])

        goon = True
        finish = False
        while goon:
            # Reflect
            for i in range(n+1):
                if i == bestind:
                    simplex_refl[:,i] = simplex[:,i]
                    fvals_refl[:,i] = fvals[:,i]
                else:
                    simplex_refl[:,i] = 2 * simplex[:,bestind] - simplex[:,i]
                    fvals_refl[:,i] = fvals[:,bestind]
            initflag = _np_false([n+1])
            initflag[bestind] = True
            fvals_refl, bestind_refl = self._find_smallest(f, simplex_refl, fvals_refl, initflag)
            if bestind_refl != bestind:
                # expand
                for i in range(n+1):
                    if i == bestind:
                        simplex_cur[:,i] = simplex_refl[:,i]
                        fvals_cur[:,i] = fvals_refl[:,i]
                    else:
                        simplex_cur[:,i] = self.expand_coef * simplex_refl[:,i] + (1 - self.expand_coef) * simplex_refl[:,bestind]
                        fvals_cur[:,i] = fvals_refl[:,i]
                subindex = _np_true([n+1])
                subindex[bestind] = False
                simplex_ex = np.concatenate([simplex_refl, simplex_cur[:,subindex]], 1)
                fvals_ex = np.concatenate([fvals_refl, fvals_cur[:,subindex]], 1)
                initflag = _np_false([2*n+1])
                initflag[:(n+1)] = _np_true([n+1])
                fvals_ex, bestind_ex = self._find_smallest(f, simplex_ex, fvals_ex, initflag)
                if bestind_ex >= n+1:
                    simplex = simplex_cur
                    fvals[:,subindex] = fvals_ex[:,(n+1):(2*n+1)]
                    fvals[:,bestind] = fvals_ex[:,bestind]
                    bestind_ex = bestind_ex - (n + 1)
                    if bestind_ex >= bestind:
                        bestind = bestind_ex + 1
                    else:
                        bestind = bestind_ex
                else:
                    simplex = simplex_refl
                    fvals = fvals_ex[:,:(n+1)]
                    bestind = bestind_ex
                goon = False
            else:
                # contract
                for i in range(n+1):
                    if i == bestind:
                        simplex_cur[:,i] = simplex[:,i]
                        fvals_cur[:,i] = fvals[:,i]
                    else:
                        simplex_cur[:,i] = self.contract_coef * simplex[:,i] + (1 - self.contract_coef) * simplex[:,bestind]
                        fvals_cur[:,i] = fvals[:,i]
                initflag = _np_false([n+1])
                initflag[bestind] = True
                fvals_cur, bestind_cur = self._find_smallest(f, simplex_cur, fvals_cur, initflag)
                if bestind_cur != bestind:
                    goon = False
                    bestind = bestind_cur
                simplex = simplex_cur
                fvals = fvals_cur
                # check stopping condition
                finish = True
                for i in range(n+1):
                    if np.linalg.norm(simplex[:,i] - simplex[:,bestind]) > self.xeps:
                        finish = False
                if finish:
                    goon = False
        return fvals, simplex, bestind, finish
