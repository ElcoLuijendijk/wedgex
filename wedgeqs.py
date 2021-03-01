"""
Functions with analytical solutions for the exhumation of an orogenic wedge
A numerical solution is also included that is based on an analytical solution of velocity fields and numerical
particle tracking using these fields.

"""

import numpy as np


def analytical_solution(t, x0, alpha, beta, L, vc, vd, vxa, vya, remove_particles_outside_wedge=True):

    """
    Analytical solution for rock particle trajectories in an orogenic wedge subject to compression, transport and
    accretion

    Parameters
    ----------
    t : array_like
        Time at which to evaluate the particle positions (s). Note that time should be negative to calculate historical
        particle trajectories
    x0 : float or array_like
        Initial horizontal position of the rock particle at t=0. Note that the particle is automatically placed at
        the land surface.
    alpha : float
        The slope of the land surface (m/m)
    beta : float
        The slope of the base of the wedge (m/m)
    L : float
        The length of the wedge (m)
    vc : float
        Horizontal compression velocity of the wedge (m/s)
    vd : float
        Horizontal transport velocity along a basal detachment (m/s)
    vxa : float
        Horizontal accretion velocity (m/s)
    vya : float
        Vertical accretion velocity (m/s)

    Returns
    -------
    x : array-like
        Horizontal positions of the particle over time
    y : array-like
        Vertical positions of the particle over time

    """

    # geometry coefficients
    epsilon = -2.0 - (beta / (alpha - beta))
    zeta = 2.0 * beta + (alpha * beta) / (alpha - beta)

    # calculate normalized compression velocity
    vn = vc / L

    # calculate transport velocity
    vxt = vd + vxa
    vyt = beta * vd + vya

    # solution for the horizontal position of rock particles over time
    x = (x0 + vxt / vn) * np.exp(vn * t) - vxt / vn

    # solution for the vertical position of rock particles over time
    y = (alpha * x0
         + beta * (x0 + vxt / vn) * (np.exp(vn * (1 - epsilon) * t) - 1)
         - ((vyt - zeta * vxt) / (vn * epsilon) * (np.exp(-vn*epsilon*t) - 1))) \
        * np.exp(vn * epsilon * t)

    # remove particles below bottom wedge
    b = beta * x
    below_domain = y <= b
    left_of_bnd = x <= 0
    right_of_bnd = x >= L
    
    if remove_particles_outside_wedge is True:
        y[below_domain] = np.nan
        x[below_domain] = np.nan
        
        x[left_of_bnd] = np.nan
        y[left_of_bnd] = np.nan
        
        x[right_of_bnd] = np.nan
        y[right_of_bnd] = np.nan

    else:
        # particles that move outside of wedge will keep their last position
        if np.any(below_domain):
            y[below_domain] = b[below_domain]
        if np.any(left_of_bnd):
            x[left_of_bnd] = 0.0
            if np.any(left_of_bnd==False):
                y[left_of_bnd] = y[left_of_bnd==False][-1]
            #else:
            #    y[left_of_bnd] = y[0]
        if np.any(right_of_bnd):
            x[right_of_bnd] = L
            if np.any(right_of_bnd==False):
                y[right_of_bnd] = y[right_of_bnd==False][-1]
            else:
                y[right_of_bnd] = y[0]

    return x, y


def analytical_solution_simplified(t, x0, alpha, beta, L, vc, vd, vxa, vya):

    """
    Simplified analytical solution for rock particle trajectories
    in an orogenic wedge subject to compression, transport and accretion

    Parameters
    ----------
    t : array_like
        Time at which to evaluate the particle positions (s). Note that time should be negative to calculate historical
        particle trajectories
    x0 : float or array_like
        Initial horizontal position of the rock particle at t=0. Note that the particle is automatically placed at
        the land surface.
    alpha : float
        The slope of the land surface (m/m)
    beta : float
        The slope of the base of the wedge (m/m)
    L : float
        The length of the wedge (m)
    vc : float
        Horizontal compression velocity of the wedge (m/s)
    vd : float
        Horizontal transport velocity along a basal detachment (m/s)
    vxa : float
        Horizontal accretion velocity (m/s)
    vya : float
        Vertical accretion velocity (m/s)

    Returns
    -------
    x : array-like
        Horizontal positions of the particle over time
    y : array-like
        Vertical positions of the particle over time

    """

    # geometry coefficient
    epsilon = -2 - (beta / (alpha - beta))

    # calculate normalized compression velocity
    vn = vc / L

    # calculate transport velocity
    vxt = vd + vxa
    vyt = beta * vd + vya

    # solution for the horizontal position of rock particles over time
    x = (x0 + vxt / vn) * np.exp(vn * t) - vxt / vn

    # solution for the depth of rock particles over time
    d = - vyt * t + (alpha - beta) * x0 * (np.exp(vn * t) - np.exp(epsilon * vn * t))

    # calculate vertical position
    y = alpha * x0 - d

    # remove particles below bottom wedge
    b = beta * x
    outside_domain = y < b
    y[outside_domain] = np.nan

    return x, y


def velocity_compression_and_transport(xs, ys, alpha, beta, L, vc, vd, vxa, vya, return_all=False):

    """
    Calculate the velocity of rock particles inside an orogenic wedge due to transport, compression and accretion

    Parameters
    ----------
    xs : array_like
        Horizontal positions where the rock particle velocity should be evaluated
    ys : float or array_like
        Vertical positions where the rock particle velocity should be evaluated
    alpha : float
        The slope of the land surface (m/m)
    beta : float
        The slope of the base of the wedge (m/m)
    L : float
        The length of the wedge (m)
    vc : float
        Horizontal compression velocity of the wedge (m/s)
    vd : float
        Horizontal transport velocity along a basal detachment (m/s)
    vxa : float
        Horizontal accretion velocity (m/s)
    vya : float
        Vertical accretion velocity (m/s)

    Returns
    -------
    vx : array-like
        Horizontal particle velocity
    vy : array-like
        Vertical particle velocity

    """

    epsilon = -2 - (beta / (alpha - beta))

    zeta = 2 * beta + (alpha * beta) / (alpha - beta)

    vn = vc / L

    vxc = np.ones_like(xs) * vn * xs

    vyc = np.ones_like(xs) * vn * (epsilon * ys + zeta * xs)

    vxt = np.ones_like(xs) * vd + vxa

    vyt = np.ones_like(xs) * beta * vd + vya

    vx = vxc + vxt

    vy = vyc + vyt

    if return_all is True:
        return vx, vy, vxc, vyc, vxt, vyt
    else:
        return vx, vy


def numerical_particle_trajectory(t, x0, alpha, beta, L, vc, vd, vxa, vya):

    """
    Numerical particle tracking for rock particle trajectories inside an orogenic wedge

    Parameters
    ----------
    t : array_like
        Time at which to evaluate the particle positions (s). Note that time should be negative to calculate historical
        particle trajectories
    x0 : float or array_like
        Initial horizontal position of the rock particle at t=0. Note that the particle is automatically placed at
        the land surface.
    alpha : float
        The slope of the land surface (m/m)
    beta : float
        The slope of the base of the wedge (m/m)
    L : float
        The length of the wedge (m)
    vc : float
        Horizontal compression velocity of the wedge (m/s)
    vd : float
        Horizontal transport velocity along a basal detachment (m/s)
    vxa : float
        Horizontal accretion velocity (m/s)
    vya : float
        Vertical accretion velocity (m/s)

    Returns
    -------
    x : array-like
        Horizontal positions of the particle over time
    y : array-like
        Vertical positions of the particle over time

    """

    xp = [x0]
    yp = [x0 * alpha]

    vx, vy = velocity_compression_and_transport(xp[-1], yp[-1], alpha, beta, L, vc, vd, vxa, vya)

    dt = np.diff(t)

    for dti in dt:
        xp.append(xp[-1] + vx * dti)
        yp.append(yp[-1] + vy * dti)

        vx, vy = velocity_compression_and_transport(xp[-1], yp[-1], alpha, beta, L, vc, vd, vxa, vya)

    x = np.array(xp)
    y = np.array(yp)

    # remove particles below bottom wedge
    b = beta * x
    outside_domain = y < b
    y[outside_domain] = np.nan

    return x, y






