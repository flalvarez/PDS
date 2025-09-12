#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 06:12:16 2025

@author: alumno
"""

import numpy as np
import matplotlib.pyplot as plt

N= 1000#
fs= 1000 #Hz
fo = 1 #DeltaF

fr=np.random.uniform(-2,2)

w0= np.pi/2
w1= w0 + fr * 2*np.pi /N
f1=  w1/2*np.pi

a0= np.sqrt(2) #raiz de 2 para tener potencia unitaria

SNR3= 3
SNR10= 10

Ps= a0**2/2 # Potencia señal

Pn10= Ps / (10**(SNR10/10)) # Potencia ruido, cuando SNR = 10 db 
sigma= np.sqrt(Pn10) #Desvio estandar
na=np.random.normal(0,sigma, N)

def mi_funcion_sen( vmax, dc, ff, ph, nn , fs):
    tt= np.arange(stop=1,step=ts)
    
    xx = vmax * np.sin(tt*2*np.pi*ff + ph) + dc

    return tt, xx
    

ts = 1/fs #s
tt, xx = mi_funcion_sen( vmax = a0, dc = 0, ff = f1, ph=0, nn = N, fs = fs/4)

SNR= 10*np.log10(np.var(xx)/np.var(na)) #Uso la varianza de xx y na porque son las señales de potencia y ruido reales, Ps y Pn son algebraicas

plt.title('Estimacion Espectral')
plt.xlabel('tiempo [s]')
plt.ylabel('V(t) [V]')
plt.plot(tt, xx, color = 'fuchsia')
