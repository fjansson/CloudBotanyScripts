import numpy as np

# linear profile => Use for u
def lin(z, u0, dudz):
    return u0 + dudz*z

# With mixed layer, where u0 is set at zml => Use for qt
def exp(z, u0, u_lambda, zml=500):
    u = u0 * np.exp(-(z-zml) / u_lambda)
    u[z<=zml] = u0
    return u

# With mixed layer with fixed offset (du0) from surface value (u0),
# and sloping FT (dudz) from fixed mixed layer height (zml)
# Use for thl
def linmlsurf(z, dudz, u0, du0=1.25, zml=500):
    u = np.zeros(z.shape)
    u[z<=zml] = u0 - du0 # Positive offsets are reductions w.r.t surface
    u[z>zml] = u0 - du0 + (z[z>zml] - zml)*dudz
    return u

# Sum of exponential dropoff and sine => Use for subsidence
def expsinw(z, w0, Hw, ampwp, Hwp):
    wbase = -w0*(1-np.exp(-z/Hw))
    wonion = ampwp*np.sin(2.*np.pi/Hwp*z)
    wonion[z>Hwp] = 0.
    return wbase + wonion