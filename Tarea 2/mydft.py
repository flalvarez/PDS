#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 07:43:55 2025

@author: alumno
"""

# Desarrollar un algoritmo que calcule la transformada discreta de Fourier (DFT).

# 𝑋(𝑘)=∑(de 𝑛=0 a n=𝑁−1) 𝑥(𝑛).𝑒−𝑗2𝜋.𝑘.𝑛/𝑁

# XX = mi_funcion_DFT( xx )

# xx: señal a analizar, una     matriz (Nx1) de números reales. 

# XX: DFT de xx, una matriz (Nx1) de números complejos.

import numpy as np
import matplotlib.pyplot as plt


N= 1000#
fs= 1000 #Hz
fo = 1 #DeltaF

ts = 1/fs #s

def mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs):
    tt = np.arange(nn) * (1/fs)   # vector de tiempo    
    xx = vmax * np.sin(tt*2*np.pi*ff + ph) + dc

    return tt, xx
   
def mi_funcion_DFT(xx):
    N=  len(xx)
    xx=np.array(xx).reshape(N, 1) #Vector columna
    n= np.arange(N).reshape(N, 1) #Armo un vector columna
    k= np.arange(N).reshape(1,N) #Armo un vector fila
    
    twiddle= np.exp(-1j*2*np.pi*n*k/N)
    
    #XX= np.dot(xx.reshape(1,N), twiddle)
    XX = np.dot(twiddle.T, xx) #El .T es para hacer la transpuesta
    return XX
    
# ts = 1/fs #s
tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 200, ph=0, nn = N, fs = fs)
XX = mi_funcion_DFT(xx)

f = np.arange(-N/2, N/2) * fs / N  #De esta forma me queda centrado y puedo ver las frecuencias posritivas y las negativa
plt.plot(f, np.abs(np.fft.fftshift(XX)), color="#2271B3") #Con el shifteo  me queda centrado y puedo ver las frecuencias posritivas y las negativa
#plt.plot(f, np.abs(XX), color= "#2271B3") #De esta forma me grafica pero en 200 y 800 Hz pero no es exactamente lo que quiero
plt.title('Transformada Discreta de Fourier')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X[k]|')


# b[n]= 1*np.exp((-1)*1j*2*np.pi*n*k/N * n)
# X[k]=np.dot(x, b)
#np.sum(x.b)

# for k in (0,N-1):
#     for n in (0,N-1):
#         X[k] += xx[n]* 1*np.exp((-1)*1j*2*np.pi*n*k/N)