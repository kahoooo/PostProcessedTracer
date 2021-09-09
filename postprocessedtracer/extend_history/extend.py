from functools import lru_cache

import numpy as np
import scipy.optimize as so
import sympy as sp
from sympy.utilities.lambdify import implemented_function


class Extend:
    def __init__(self, sympy_func_m_i, sympy_func_m_f, sympy_func_r, dt, mmax, omega_i, steps):
        def pre_lambidify(expr):
            return sp.simplify(sp.nfloat(expr), doit=False)

        def floatify(expr, n=15):
            from sympy.core.rules import Transform
            return expr.xreplace(Transform(
                lambda x: sp.Float(x.n(n), n),
                lambda x: x.is_number))

        def nintegrate(f, var):
            if isinstance(var, sp.Symbol):
                x = var
            elif isinstance(var, tuple):
                x = var[0]
            else:
                raise ValueError(f'unexpected {var = }')

            def recursive(f_):
                if type(f_) == sp.Add:
                    return sp.Add(*map(recursive, f_.args))
                if type(f_) == sp.Mul:
                    outside = sp.Mul(*(arg for arg in f_.args if arg.free_symbols and not arg.has(x)))
                    inside = sp.Mul(*(arg for arg in f_.args if not arg.has(t) or arg.has(x)))
                    return outside * sp.Integral(inside, var).n()
                return sp.Integral(f_, var).n()

            return recursive(f)

        # retrieve the mass distribution and intermediate position expressions
        m, t = sp.symbols('m, t', real=True)
        r_i = sp.Function('r_i', real=True)(m)
        r_f = sp.Function('r_f', real=True)(m)
        m_i = sympy_func_m_i(r_i)
        m_f = sympy_func_m_f(r_f)
        r = sympy_func_r(t, dt, r_i, r_f)

        # first derivatives
        vr = r.diff(t)
        dmi_dri = m_i.diff(r_i)
        dmf_drf = m_f.diff(r_f)
        dm_dr = 1 / r.diff(m).subs({r_i.diff(m): 1 / dmi_dri, r_f.diff(m): 1 / dmf_drf})
        # dri_drf = dmf_drf / dmi_dri
        # drf_dri = dmi_dri / dmf_drf

        # second derivatives
        d2mi_dri2 = dmi_dri.diff(r_i)
        d2mf_drf2 = dmf_drf.diff(r_f)
        # d2ri_drf2 = (dmi_dri * d2mf_drf2 - dmf_drf * dri_drf * d2mi_dri2) / dmi_dri ** 2
        # d2rf_dri2 = (dmf_drf * d2mi_dri2 - dmi_dri * drf_dri * d2mf_drf2) / dmf_drf ** 2

        # in general, m_i and m_f are not invertible analytically, solve numerically
        lamb_m_i_m = sp.lambdify((r_i, m), pre_lambidify(m_i - m))
        lamb_m_f_m = sp.lambdify((r_f, m), pre_lambidify(m_f - m))
        lamb_m_i_m_prime = sp.lambdify((r_i, m), pre_lambidify(dmi_dri))
        lamb_m_f_m_prime = sp.lambdify((r_f, m), pre_lambidify(dmf_drf))
        lamb_m_i_m_prime2 = sp.lambdify((r_i, m), pre_lambidify(d2mi_dri2))
        lamb_m_f_m_prime2 = sp.lambdify((r_f, m), pre_lambidify(d2mf_drf2))

        @lru_cache
        def r_i_nsolve(m_):
            m_ = float(m_)
            return so.root_scalar(lamb_m_i_m,
                                  fprime=lamb_m_i_m_prime,
                                  fprime2=lamb_m_i_m_prime2,
                                  args=(m_,), x0=1.0, method='halley').root

        @lru_cache
        def r_f_nsolve(m_):
            m_ = float(m_)
            return so.root_scalar(lamb_m_f_m,
                                  fprime=lamb_m_f_m_prime,
                                  fprime2=lamb_m_f_m_prime2,
                                  args=(m_,), x0=1.0, method='halley').root

        r_i_impl = implemented_function('r_i_impl', r_i_nsolve)(m)
        r_f_impl = implemented_function('r_f_impl', r_f_nsolve)(m)

        lamb_m_i = sp.lambdify(r_i, pre_lambidify(m_i))
        lamb_m_f = sp.lambdify(r_f, pre_lambidify(m_f))
        lamb_r = sp.lambdify((t, m), pre_lambidify(r),
                             [{'r_i': r_i_nsolve, 'r_f': r_f_nsolve}, 'numpy', 'scipy'])
        lamb_vr = sp.lambdify((t, m), pre_lambidify(vr),
                              [{'r_i': r_i_nsolve, 'r_f': r_f_nsolve}, 'numpy', 'scipy'])

        # solve density in mass coordinates, moment of inertia and angular speed as a function of time, and lambdify
        rho = dm_dr / (4 * sp.pi * r ** 2)
        lamb_rho = sp.lambdify((t, m), pre_lambidify(rho),
                               [{'r_i': r_i_nsolve, 'r_f': r_f_nsolve}, 'numpy', 'scipy'])

        moi_integrand = (2 / 3 * rho * 4 * sp.pi * r ** 4 / dm_dr).subs({r_i: r_i_impl, r_f: r_f_impl})
        moi = nintegrate(moi_integrand.expand(), (m, 0, mmax))
        lamb_moi = sp.lambdify(t, pre_lambidify(moi),
                               [{'r_i': r_i_nsolve, 'r_f': r_f_nsolve}, 'numpy', 'scipy'])

        omega = omega_i * lamb_moi(0) / moi
        lamb_omega = sp.lambdify(t, pre_lambidify(omega),
                                 [{'r_i': r_i_nsolve, 'r_f': r_f_nsolve}, 'numpy', 'scipy'])

        # finally, integrate and get angular displacement
        t_s, t_e = sp.symbols('t_s, t_e', real=True)
        phi = sp.integrate(sp.nsimplify(omega), (t, t_s, t_e), heurisch=True)
        lamb_phi = sp.lambdify((t_s, t_e), floatify(pre_lambidify(phi)),
                               [{'r_i': r_i_nsolve, 'r_f': r_f_nsolve}, 'numpy', 'scipy'])

        time_arr = np.linspace(0, dt, num=steps + 1)
        omega_arr = np.vectorize(lamb_omega)(time_arr)
        dphi_arr = np.vectorize(lamb_phi)(time_arr[:-1], time_arr[1:])
        phi_arr = np.zeros_like(time_arr)
        phi_arr[1:] = np.cumsum(dphi_arr)

        self.time = time_arr
        self.lamb_m_i = lamb_m_i
        self.lamb_m_f = lamb_m_f
        self.lamb_r = lamb_r
        self.lamb_vr = lamb_vr
        self.lamb_rho = lamb_rho
        self.omega = omega_arr
        self.phi = phi_arr

    def history(self, r_i=None, r_f=None):
        if r_i is not None:
            m = self.lamb_m_i(r_i)
        elif r_f is not None:
            m = self.lamb_m_f(r_f)
        else:
            raise TypeError('either r_i or r_f has to be provided')
        time = self.time
        r = np.vectorize(self.lamb_r)(time, m)
        phi = self.phi
        vr = np.vectorize(self.lamb_vr)(time, m)
        vphi = r * self.omega
        rho = np.vectorize(self.lamb_rho)(time, m)

        return time, r, phi, vr, vphi, rho
