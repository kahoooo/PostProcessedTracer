import sympy as sp


def uniform_sphere(rho_0=None):
    if rho_0 is None:
        raise TypeError(f'{uniform_sphere.__name__}() requires arguments rho_0')

    def func(r):
        return 4 * sp.pi / 3 * rho_0 * r ** 3

    return func


def singular_isothermal_sphere(a=None, m_0=None, r_0=None):
    if a is None == (m_0 is None or r_0 is None):
        raise TypeError(f'{singular_isothermal_sphere.__name__}() requires arguments either A or (M_0, r_0)')
    if a is None:
        a = m_0 / r_0

    def func(r):
        return a * r

    return func


def pseudo_bonnor_ebert_sphere(rho_c=None, r_c=None):
    if rho_c is None or r_c is None:
        raise TypeError(f'{pseudo_bonnor_ebert_sphere.__name__}() requires arguments (rho_c, r_c)')

    def func(r):
        return sp.integrate(rho_c / (1 + (r / r_c) ** 2) * 4 * sp.pi * r ** 2, (r, 0, r))

    return func


def constant_velocity():
    def func(t, t_c, r_i, r_f):
        return r_i + (r_f - r_i) * t / t_c

    return func


def tanh_velocity(eps=None):
    if eps is None:
        raise TypeError(f'{tanh_velocity.__name__}() requires arguments eps')

    def func(t, t_c, r_i, r_f):
        norm = (r_f - r_i) / sp.integrate(sp.tanh(t / (eps * t_c)), (t, 0, t_c))
        return r_i + sp.integrate(sp.tanh(t / (eps * t_c)), (t, 0, t)) * norm

    return func


def inspect_lambda(func):
    from inspect import findsource
    return ''.join(findsource(func)[0])
