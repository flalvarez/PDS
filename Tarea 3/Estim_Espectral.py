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
res_esp= fs/N  # DeltaF (Resolucion espectral)
M = 200        # Número de realizaciones
ts = 1/fs      # Periodo de muestreo

# Frecuencias aleatorias 
fr = np.random.uniform(-2, 2, size=M) 
w0 = np.pi/2 
w1 = w0 + fr * 2*np.pi / N 
f1 = w1/(2*np.pi) * fs #Cuando w1=pi/2 esto es fs/4 
a0 = np.sqrt(2) # para tener potencia unitaria 
# Ruido 
SNR10 = 10 
Ps = a0**2 / 2 
Pn10 = Ps / (10**(SNR10/10)) 
sigma = np.sqrt(Pn10) 
na = np.random.normal(0, sigma, size=(M, N)) # ruido (M x N) 

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs 
    tt = np.arange(0, 1, ts)[:nn] 
    xx = vmax * np.sin(2*np.pi*ff*tt + ph) + dc 
    return tt, xx 

tt, xx = mi_funcion_sen(vmax=a0, dc=0, ff=1, ph=0, nn=N, fs=fs) 
SNR_medido= 10*np.log10(np.var(xx)/np.var(na)) #Uso la varianza de xx y na porque son las señales de potencia y ruido reales, Ps y Pn son algebraicas 

# 200 realizaciones (M x N) => Realizaciones x Frecuencias (la que quiero + el ruido) 
tt_matriz= np.tile(tt.reshape((1,N)), (M,1))
f1_matriz = np.tile(f1.reshape((M,1)), (1,N))

realizaciones_xx = a0 * np.sin(2*np.pi *f1_matriz * tt_matriz) + na 

# -------------------------------------- 
# Ventanas 
# -------------------------------------- 

# Armo las ventanas 
v_ft = np.resize(windows.flattop(N), (M,N)) 
v_bh = np.resize(windows.blackmanharris(N), (M,N)) 
v_h = np.resize(windows.hamming(N), (M,N)) 

# Aplico las ventanas 
realizaciones_xx_ft = realizaciones_xx * v_ft 
realizaciones_xx_bh = realizaciones_xx * v_bh 
realizaciones_xx_h = realizaciones_xx * v_h 

XX = np.fft.fft(realizaciones_xx)
XX_ft = np.fft.fft(realizaciones_xx_ft)
XX_bh = np.fft.fft(realizaciones_xx_bh)
XX_h = np.fft.fft(realizaciones_xx_h)

# Concateno todas las realizaciones 
# concat_XX = np.hstack(XX) 
# concat_XX_ft = np.hstack(XX_ft) 
# concat_XX_bh = np.hstack(XX_bh) 
# concat_XX_h = np.hstack(XX_h) 

# Frecuencia para el eje x 
f = np.linspace(0, fs, N) 
ff= np.tile(f.reshape((1,N)), (M,1))
f_concat = np.tile(f, M) 
w = 2 * np.pi * f / fs

 # Ganancia de cada ventana. Divido por N para normalizar 
G_ft = np.sum(windows.flattop(N)) / N 
G_bh = np.sum(windows.blackmanharris(N)) / N 
G_h = np.sum(windows.hamming(N)) / N 
 
#axis=None (default): Calculates the mean of the flattened array. 
# axis=0: Calculates the mean along the columns. 
# axis=1: Calculates the mean along the rows. 

# Cada columna corresponde a una frecuencia específica k. 
# Cada fila corresponde a una realización diferente. 
#Uso axis = 1 

# Sin ventana 
idx_max_sin_vent = np.argmax(np.abs(XX), axis=1) # índice del pico de cada realización 
est_amp_sin_vent = 2 * np.abs(XX[np.arange(M), idx_max_sin_vent])/N #El *2 es porque se reparte mitad para las frecuencias positivas y mitad para las frecuencias negativas (como lo vimos en comu) 

# Flattop 
idx_max_ft = np.argmax(np.abs(XX_ft), axis=1) 
est_amp_ft = 2 * np.abs(XX_ft[np.arange(M), idx_max_ft]) / (N * G_ft) 

# Blackman-Harris 
idx_max_bh = np.argmax(np.abs(XX_bh), axis=1) 
est_amp_bh = 2 * np.abs(XX_bh[np.arange(M), idx_max_bh]) / (N * G_bh) 

# Hamming 
idx_max_h = np.argmax(np.abs(XX_h), axis=1) 
est_amp_h = 2 * np.abs(XX_h[np.arange(M), idx_max_h]) / (N * G_h) 

## 
# Estimador de amplitud 
## 
#Hago una funcion para no tener que hacerlo para los 4 casos 
def Estadistica (est_amp, a0): 
    E_a = np.mean(est_amp) 
    S_a = E_a - a0 
    V_a = np.mean((est_amp - E_a)**2) 
    return E_a, S_a, V_a 

E, S, V = Estadistica(est_amp_sin_vent, a0) 
E_ft, S_ft, V_ft = Estadistica(est_amp_ft, a0) 
E_bh, S_bh, V_bh = Estadistica(est_amp_bh, a0) 
E_h, S_h, V_h = Estadistica(est_amp_h, a0) 

print("Amplitud media:", E)
print("Sesgo:", S) 
print("Varianza:", V) 


freqs = np.fft.fftfreq(N, d=1/fs)   # eje FFT coherente [-fs/2, fs/2)
## 
# Estimador de frecuencia 
## 
est_frec_sin_vent = np.abs(freqs[idx_max_sin_vent])
est_frec_ft = np.abs(freqs[idx_max_ft])
est_frec_bh = np.abs(freqs[idx_max_bh])
est_frec_h = np.abs(freqs[idx_max_h])

def EstadisticaFrec(est_frec, f_ref): 
    E_f = np.mean(est_frec) 
    S_f = E_f - f_ref 
    V_f = np.var(est_frec) 
    return E_f, S_f, V_f 

f_ref = np.mean(f1) 
E_f, S_f, V_f = EstadisticaFrec(est_frec_sin_vent, f_ref) 
E_f_ft, S_f_ft, V_f_ft = EstadisticaFrec(est_frec_ft, f_ref) 
E_f_bh, S_f_bh, V_f_bh = EstadisticaFrec(est_frec_bh, f_ref) 
E_f_h, S_f_h, V_f_h = EstadisticaFrec(est_frec_h, f_ref) 

print("Estimador de Frecuencia:", E_f) 
print("Sesgo de frecuencia:", S_f) 
print("Varianza de la frecuencia:", V_f) 

plt.figure() 
plt.plot(ff, 20*np.log10(np.abs(XX)), 'x', color='hotpink') #en dB 
#plt.plot(f_concat, np.abs(concat_XX), color='hotpink') #En veces
plt.xlim(0, 500) # Limita el eje x 
plt.title('Magnitud promedio sin ventana') 
plt.xlabel('Frecuencia [Hz]') 
plt.ylabel('Amplitud [V]')

plt.figure() 
plt.plot(ff, 20*np.log10(np.abs(XX_ft)), 'x', color='darkturquoise') 
#plt.plot(f_concat, np.abs(concat_XX_ft), color='darkturquoise') 
plt.xlim(0, 500) # Limita el eje x
plt.title('Magnitud promedio con ventana Flattop') 
plt.xlabel('Frecuencia [Hz]') 
plt.ylabel('Amplitud [V]') 

plt.figure() 
plt.plot(ff, 20*np.log10(np.abs(XX_bh)), 'x', color='darkblue')
#plt.plot(f_concat, np.abs(concat_XX_bh), color='darkblue') 
plt.xlim(0, 500) # Limita el eje x
plt.title('Magnitud promedio con ventana Blackman-Harris') 
plt.xlabel('Frecuencia [Hz]') 
plt.ylabel('Amplitud [V]') 

plt.figure() 
plt.plot(ff, 20*np.log10(np.abs(XX_h)), 'x', color='darkmagenta') 
#plt.plot(f_concat, np.abs(concat_XX_h), color='darkmagenta') 
plt.xlim(0, 500) # Limita el eje x
plt.title('Magnitud promedio con ventana Hamming') 
plt.xlabel('Frecuencia [Hz]') 
plt.ylabel('Amplitud [V]') 

plt.figure()
plt.hist(est_amp_sin_vent, bins=30, histtype='step', label="Sin ventana", color='hotpink')
plt.hist(est_amp_ft, bins=30, histtype='step', label="Flattop", color='darkturquoise')
plt.hist(est_amp_bh, bins=30, histtype='step', label="Blackman-Harris", color='darkblue')
plt.hist(est_amp_h, bins=30, histtype='step', label="Hamming", color='darkmagenta')
plt.xlabel("Amplitud estimada")
plt.ylabel("Número de realizaciones")
plt.title("Histograma de amplitud estimada")
plt.legend()
plt.show()

plt.figure()
plt.hist(est_frec_sin_vent, bins=30, histtype='step', label="Sin ventana", color='hotpink', linestyle="--", linewidth=2)
plt.hist(est_frec_ft, bins=30, histtype='step', label="Flattop", color='darkturquoise', linestyle="--", linewidth=1.5)
plt.hist(est_frec_bh, bins=30, histtype='step', label="Blackman-Harris", color='darkblue', linestyle="--", linewidth=1.5)
plt.hist(est_frec_h, bins=30, histtype='step', label="Hamming", color='darkmagenta', linestyle="--", linewidth=1.5)
plt.xlabel("Frecuencia estimada")
plt.ylabel("Número de realizaciones")
plt.title("Histograma de frecuencia estimada")
plt.legend()
plt.show()
