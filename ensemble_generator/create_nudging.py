#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:47:20 2022

@author: martinjanssens
"""

import numpy as np
# import matplotlib.pyplot as plt
import os

def _nudge_atan(x, a=5, b=2, c=20, lev_max_change=3000, end=3600*6):
    """	
    Nudging timescale creation, by Alessandro Savazzi. Free-tropospheric
    nudging time set by end parameter (default is 6 hours), with the max
    gradient in nudging time occurring at lev_max_change.
    """

    y = b * (np.pi/2+np.arctan(a* np.pi/2*(1-x/lev_max_change)))
    y = end + y**c
    # plot
    # plt.figure(figsize=(6,9))
    # plt.plot(y,x)
    # plt.xlim([10e3,10e8])
    # plt.ylim([0,5000])
    # plt.xscale('log')
    return y

def create_nudging(zf, thl, qt, u, v, nudge_params, out_dir):
    """
    Makes and writes a nudging input file to out_dir/nudge.inp.001, based on 
    profiles to nudge towards of
    - zf
    - thl
    - qt
    - u
    - v
    nudge_params is a tuple that contains the input parameters to Alessandro's 
    arctangent nudging function.
    """
    
    zero = np.zeros(zf.shape)
    (a,b,c,z_max_change,tnudge_ft) = nudge_params
    
    # Nudging factor with height; 
    # is multiplied with nudging time (tnudgefac) from namelist;
    # here we set tnudgefac=1 -> Then this is the nudging time in seconds
    nudgefac = _nudge_atan(zf,a,b,c,z_max_change,tnudge_ft)
    
    out_profs = np.stack((zf,nudgefac,u,v,zero,thl,qt)).T
    nudge_out = os.path.join(out_dir, 'nudge.inp.001')
    
    # Empty
    f = open(nudge_out,'w')
    f.close()
    
    # And append two time instances - one at start, one after end of simulation
    with open(nudge_out, 'ab') as f:
        np.savetxt(f, out_profs, fmt='%+10.10e', comments='',
                   header='\n      z (m)          factor (-)         u (m s-1)         v (m s-1)         w (m s-1)          thl (K)        qt (kg kg-1)    \n# 0.00000000E+00')
        np.savetxt(f, out_profs, fmt='%+10.10e', comments='',
                   header='\n      z (m)          factor (-)         u (m s-1)         v (m s-1)         w (m s-1)          thl (K)        qt (kg kg-1)    \n# 1.00000000E+07')

if __name__ == '__main__':
    
    lp = os.getcwd()
    tnudge_ft = 6 # hours
    
    nudge_params = (2,3,7.4,3000,tnudge_ft*3600)
    
    # Vertical profiles => Nudge to initial state
    target_profs = np.loadtxt(lp+'/prof.inp.001')
    zf = target_profs[:,0]
    thl = target_profs[:,1]
    qt = target_profs[:,2]
    u = target_profs[:,3]
    v = target_profs[:,4]
    
    create_nudging(zf, thl, qt, u, v, nudge_params, lp)
    
