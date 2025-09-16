#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 06:12:16 2025

@author: alumno
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as windows

N = 1000       # Número de muestras
fs = 1000      # Hz
res_esp= fs/N  # DeltaF
M = 200        # Número de realizaciones
ts = 1/fs      # Periodo de muestreo

# Frecuencias aleatorias
fr = np.random.uniform(-2, 2, size=M)
w0 = np.pi/2
w1 = w0 + fr * 2*np.pi / N
f1 = w1/(2*np.pi) * fs   #Cuando w1=pi/2 (no es aleatorio) esto es fs/4

a0 = np.sqrt(2)   # para tener potencia unitaria

# Ruido
SNR10 = 10
Ps = a0**2 / 2
Pn10 = Ps / (10**(SNR10/10))
sigma = np.sqrt(Pn10)
na = np.random.normal(0, sigma, size=(M, N))  # ruido (M x N)


def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs
    tt = np.arange(0, 1, ts)[:nn]
    xx = vmax * np.sin(2*np.pi*ff*tt + ph) + dc
    return tt, xx

def mi_funcion_DFT(xx):
    N = len(xx)
    xx = np.array(xx).reshape(N,1)  # vector columna
    n = np.arange(N).reshape(N,1)
    k = np.arange(N).reshape(1,N)
    twiddle = np.exp(-1j*2*np.pi*n*k/N)
    XX = np.dot(twiddle.T, xx)
    return XX


tt, xx = mi_funcion_sen(vmax=a0, dc=0, ff=1, ph=0, nn=N, fs=fs)
SNR_medido= 10*np.log10(np.var(xx)/np.var(na)) #Uso la varianza de xx y na porque son las señales de potencia y ruido reales, Ps y Pn son algebraicas

# 200 realizaciones (M x N) => Realizaciones x Frecuencias (la que quiero + el ruido)
realizaciones_xx = a0 * np.sin(2*np.pi * f1[:,None] * tt[None,:]) + na # Lo de  f1[:, None] funciona como el resize pero ocupa menos memoria, crea un vector columna (Mx1)

# --------------------------------------
# Ventanas
# --------------------------------------

# Armo las ventanas
v_ft = np.resize(windows.flattop(N), (M,N))
v_bh = np.resize(windows.blackmanharris(N), (M,N))
v_h  = np.resize(windows.hamming(N), (M,N))

# Aplico las ventanas
realizaciones_xx_ft = realizaciones_xx * v_ft
realizaciones_xx_bh = realizaciones_xx * v_bh
realizaciones_xx_h  = realizaciones_xx * v_h

# Transformo a frecuencia. Lo hago asi porque cuando quise usar mi funcion se rompia todo, no entiendo por que si es lo mismo
n = np.arange(N).reshape(N,1)
k = np.arange(N).reshape(1,N)
twiddle = np.exp(-1j*2*np.pi*n*k/N)

XX    = np.dot(realizaciones_xx, twiddle.T)
XX_ft = np.dot(realizaciones_xx_ft, twiddle.T)
XX_bh = np.dot(realizaciones_xx_bh, twiddle.T)
XX_h  = np.dot(realizaciones_xx_h, twiddle.T)

#Magnitudes para el eje y
# axis=None (default): Calculates the mean of the flattened array.
# axis=0: Calculates the mean along the columns.
# axis=1: Calculates the mean along the rows.

# Cada columna corresponde a una frecuencia específica k.
# Cada fila corresponde a una realización diferente.
#Uso axis = 0 
mag    = 20*np.log10(np.mean(np.abs(XX), axis=0)) 
mag_ft = 20*np.log10(np.mean(np.abs(XX_ft), axis=0))
mag_bh = 20*np.log10(np.mean(np.abs(XX_bh), axis=0))
mag_h  = 20*np.log10(np.mean(np.abs(XX_h), axis=0))


# Frecuencia para el eje x
f = np.linspace(0, fs, N)

plt.figure()
plt.plot(f, mag, color='hotpink')
plt.title('Magnitud promedio sin ventana')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X[k]| [dB]')

plt.figure()
plt.plot(f, mag_ft, color='darkturquoise')
plt.title('Magnitud promedio con ventana Flattop')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X[k]| [dB]')

plt.figure()
plt.plot(f, mag_bh, color='darkblue')
plt.title('Magnitud promedio con ventana Blackman-Harris')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X[k]| [dB]')

plt.figure()
plt.plot(f, mag_h, color='darkmagenta')
plt.title('Magnitud promedio con ventana Hamming')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X[k]| [dB]')

plt.show()

