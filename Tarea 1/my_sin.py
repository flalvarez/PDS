#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 07:43:35 2025

@author: alumno
"""

import numpy as np
import matplotlib.pyplot as plt

N= 1000#
fs= 1000 #Hz
fo = 500 #DeltaF
   


def mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs):
    tt= np.arange(stop=1,step=ts)
    
    xx = vmax * np.sin(tt*2*np.pi*ff + ph) + dc

    return tt, xx
    

ts = 1/fs #s

tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 2, ph=0, nn = N, fs = fs)

plt.title('Funcion senoidal f=2 [Hz]')
plt.xlabel('tiempo [s]')
plt.ylabel('V(t) [V]')
plt.plot(tt, xx, color = 'fuchsia')
