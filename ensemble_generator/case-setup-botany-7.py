#!/usr/bin/env python


import numpy as np
from scipy.interpolate import interp1d
import sys
import os
import f90nml
import matplotlib.pyplot as plt
import profiles_botany7 as prof
from create_nudging import create_nudging

namoptions = sys.argv[1]

if len(sys.argv) > 2:
    out_dir = sys.argv[2]
else:
    out_dir = './'


nml = f90nml.read(namoptions)

nsv = nml['RUN']['nsv']
case = nml['VVUQ_extra']['case']

#intrad = False
#if nml['PHYSICS']['iradiation'] != 0:
#    intrad = True

#stretch = False
#dz =  25 # m
#Nz = 200 # 5 km high
Nz = nml['DOMAIN']['kmax']
stretch = True
alpha = 1.01  #  dz = 15m at surface, 30m @ 1500m
Nz = 175      #  148: top at 5016   175: top at 7km
dz = 15       #

qt_tend = 0 # large-scale qt tendency


if not stretch:
    # equally spaced levels, first level is 1/2 dz over ground
    z = (np.arange(Nz) +.5) * dz
else:
    # stretched vertical levels. First level is of size dz and located at dz/2, dz then stretched by factor alpha
    z = np.zeros(Nz)
    z[0] = dz/2
    dz_ = dz
    for i in range(1,Nz):
        z[i] = z[i-1] + dz_
        dz_ *= alpha

#total height:
if stretch and alpha > 1:
    Ztot = dz * (1-alpha**Nz) / (1-alpha)
else:
    Ztot = dz*Nz

#print(f"Nz = {Nz}, total height = {Ztot}")


# note: Fortran name lists are case-insensitive
# when using .get(), use lower-case name
# in the name list it's written as thl_Gamma, but that is not recognized.
Gamma = nml['VVUQ_extra'].get('thl_gamma', 1) # K / km
Gamma /= 1000 # convert to K/m

qt0 = nml['VVUQ_extra'].get('qt0', 0.016) # kg / kg
qt_lambda = nml['VVUQ_extra'].get('qt_lambda', 1500) # m

# optional settings for an initial mixed layer.
# thl and qt are constant in the mixed layer, and linearly interpolated to the ft value at z_ft
z_ml = nml['VVUQ_extra'].get('z_ml', 0) # height of initial mixed layer
z_ft = nml['VVUQ_extra'].get('z_ft', z_ml) # start of free troposphere (with exponential qt profile)
qt_ml = nml['VVUQ_extra'].get('qt_ml', qt0) # qt in the mixed layer
thl_ml = nml['VVUQ_extra'].get('thl_ml', None) # thl in the mixed layer

# replace wind_high by dudz, wind_low by u0  dudz = 0.0022 m/s/m
thls = nml['PHYSICS'].get('thls', None) # sea surface thl
dthl0 = nml['VVUQ_extra'].get('dthl0', None) # surface thl offset from thls
u0 = nml['VVUQ_extra'].get('u0', None) # surface wind
dudz = nml['VVUQ_extra'].get('dudz', None)  # wind shear

w0 = nml['VVUQ_extra']['w0']   # default 4e-3 m/s = 0.4 cm/s  (positive number for downward motion)
wpamp = nml['VVUQ_extra']['wpamp'] # sine-shape amplitude in subsidence

# Subsidence - exponential profile
Hw  = 2500  # m   subsidence height scale (was 1000 for botany-5)
Hwp = 5300  # m

qt = prof.exp(z, qt0, qt_lambda, z_ml)
thl = prof.linmlsurf(z, Gamma, thls, dthl0, z_ml)
u = prof.lin(z, u0, dudz)
v = np.zeros_like(z)
w_subs = prof.expsinw(z, w0, Hw, wpamp, Hwp)


# tke
tke_z      = np.array([     0,    4000,   5000]) # like RICO
tke_points = np.array([     1,    1e-8,   1e-8])

# thl and qt tendencies. Surface value specified, linearly decays to 0 at a fixed height.
thl_tend_0     = nml['VVUQ_extra'].get('thl_tend0', 0)        # surface tendency value
qt_tend_0      = nml['VVUQ_extra'].get('qt_tend0', 0)
thl_tend_z_max = nml['VVUQ_extra'].get('thl_tend_z_max', 2000) # z where tendency has dropped to 0
qt_tend_z_max  = nml['VVUQ_extra'].get('qt_tend_z_max',  4000)

thl_tend_z      = np.array([         0, thl_tend_z_max, 20000])
thl_tend_points = np.array([thl_tend_0,              0,     0])

qt_tend_z      = np.array([        0,   qt_tend_z_max,  20000])
qt_tend_points = np.array([qt_tend_0,               0,      0])

thl_tend_fun = interp1d(thl_tend_z,  thl_tend_points, fill_value=0)
qt_tend_fun  = interp1d(qt_tend_z,   qt_tend_points,  fill_value=0)
tke_fun = interp1d(tke_z,  tke_points, fill_value="extrapolate")

dthlrad = thl_tend_fun(z)
qt_tend = qt_tend_fun(z)

tke = tke_fun(z)

# write DALES input files
# print(z.shape, thl.shape, qt.shape, u.shape, v.shape, tke.shape)
newprof = np.stack((z, thl, qt, u, v, tke), axis=1)
# height  th_l  q_t   u    v  TKE
profile_out = os.path.join(out_dir, 'prof.inp.001')
np.savetxt(profile_out, newprof, fmt='%12.6g',
           header='\n    height         th_l          q_t            u            v          TKE')

# lscale.inp.nnn
# height      ug        vg       wfls      dqtdxls  dqtdyls  dqtdtls dthlrad
lscale = np.zeros((Nz, 8))
lscale[:,0] = z
lscale[:,1] = u
lscale[:,2] = v
lscale[:,3] = w_subs
lscale[:,6] = qt_tend
lscale[:,7] = dthlrad

lscale_out = os.path.join(out_dir, 'lscale.inp.001')
np.savetxt(lscale_out, lscale, fmt='%12.6g',
header='\n    height           ug           vg         wfls      dqtdxls      dqtdyls      dqtdtls      dthlrad')


#  Scalar Profiles - all 0 for now
#  height  sv0  sv1 ...
if nsv > 0:
    scalars = np.zeros((Nz, nsv+1))
    scalars[:,0] = z
    scalars_out = os.path.join(out_dir, 'scalar.inp.001')
    np.savetxt(scalars_out, scalars, fmt='%12.6g', header='\n    height          sv1          sv2 ...')

tnudge_ft = 6 # hours
nudge_params = (2,3,7.4,3000,tnudge_ft*3600)
create_nudging(z, thl, qt, u, v, nudge_params, './')
