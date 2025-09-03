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
delta= fs/N

desp= fs/4 + delta/2 

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
# tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 200, ph=0, nn = N, fs = fs)
# tt2, xx2 = mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs)
# tt2, xx3 = mi_funcion_sen( vmax = 1, dc = 0, ff = 1.5, ph=0, nn = N, fs = fs)
# tt3, xx4 = mi_funcion_sen( vmax = 1, dc = 0, ff = 2, ph=0, nn = N, fs = fs)
# tt, xx5 = mi_funcion_sen( vmax = 1, dc = 0, ff = -1, ph=0, nn = N, fs = fs)
# tt, xx6 = mi_funcion_sen( vmax = 1, dc = 0, ff = 450, ph=0, nn = N, fs = fs)
# tt, xx7 = mi_funcion_sen( vmax = 1, dc = 0, ff = 500, ph=0, nn = N, fs = fs)
# xx8= 3*xx6+2*xx7 
tt9, xx9 = mi_funcion_sen( vmax = 1, dc = 0, ff = 250, ph=0, nn = N, fs = fs)
tt10, xx10 = mi_funcion_sen( vmax = 1, dc = 0, ff =desp , ph=0, nn = N, fs = fs)


# XX = mi_funcion_DFT(xx)
# XX2 = mi_funcion_DFT(xx2)
# XX3 = mi_funcion_DFT(xx3)
# XX4 = mi_funcion_DFT(xx4)
# XX5 = mi_funcion_DFT(xx5)
# XX6 = mi_funcion_DFT(xx6)
# XX7 = mi_funcion_DFT(xx7)
# XX8 = mi_funcion_DFT(xx8) #transformada de la suma
XX9 = mi_funcion_DFT(xx9)
XX10 = mi_funcion_DFT(xx10)

#XXs = 3*XX6 + 2*XX7 #suma de las transferencias


f = np.arange(0, N) * fs / N 
# f = np.arange(-N/2, N/2) * fs / N  #De esta forma me queda centrado y puedo ver las frecuencias posritivas y las negativa
#plt.plot(f, np.abs(np.fft.fftshift(XX)), color="#2271B3") #Con el shifteo  me queda centrado y puedo ver las frecuencias posritivas y las negativa
#plt.plot(f, np.abs(XX), color= "#2271B3") #De esta forma me grafica pero en 200 y 800 Hz pero no es exactamente lo que quiero


# plt.plot(f, np.abs(XX), color = 'fuchsia') 
# plt.plot(f, np.abs(XX2), color = 'blue')
# plt.plot(f, np.abs(XX3), 'x', color = 'green')
# plt.plot(f, np.abs(XX4), color = 'red')
# plt.plot(f, np.abs(XX5), color = 'purple')
# plt.plot(f, np.abs(XX6), color = 'maroon')
# plt.plot(f, np.abs(XX7), color = 'orange')
# plt.plot(f, np.abs(XX8), 'o')
# plt.plot(f, np.abs(XXs), 'x')

plt.figure()
plt.plot( 20*np.log10(np.abs(XX9)),'o', color = 'orange')
plt.figure()
plt.plot( 20*np.log10(np.abs(XX10)),'o', color = 'fuchsia')

plt.show()

# plt.title('Transformada Discreta de Fourier')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('|X[k]|')

# plt.plot(f, np.abs(10.0 * np.log10(XX9)),'o', color = 'orange')
# plt.plot(f, np.abs(10.0 * np.log10(XX9)),'x', color = 'fuchsia')
# plt.title('Transformada Discreta de Fourier')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('|X[k]| [db]')



# b[n]= 1*np.exp((-1)*1j*2*np.pi*n*k/N * n)
# X[k]=np.dot(x, b)
#np.sum(x.b)

# for k in (0,N-1):
#     for n in (0,N-1):
#         X[k] += xx[n]* 1*np.exp((-1)*1j*2*np.pi*n*k/N)