#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 06:12:16 2025

@author: alumno
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as windows

N= 1000#
fs= 1000 #Hz
res_esp= fs/N #DeltaF
M= 200 #Numero de realizaciones 

fr=np.random.uniform(-2,2)
w0= np.pi/2
w1= w0 + fr * 2*np.pi /N
f1=  w1/2*np.pi *fs #Cuando w0=pi/2 esto es fs/4

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
    
def mi_funcion_DFT(xx):
    N=  len(xx)
    xx=np.array(xx).reshape(N, 1) #Vector columna
    n= np.arange(N).reshape(N, 1) #Armo un vector columna
    k= np.arange(N).reshape(1,N) #Armo un vector fila
    
    twiddle= np.exp(-1j*2*np.pi*n*k/N)
    
    XX = np.dot(twiddle.T, xx) #El .T es para hacer la transpuesta
    return XX

ts = 1/fs #s

# Ventanas

#Sin ventana
tt, xx = mi_funcion_sen( vmax = a0, dc = 0, ff = f1, ph=0, nn = N, fs = fs) #Esta es la señal que genero

#Flattop
v_ft = windows.flattop(N)

#Blackman-Harris
v_bh = windows.blackmanharris(N) 

#Hamming
v_h = windows.hamming(N)


# Paso al dominio de la frecuencia y agrego las ventanas
X = mi_funcion_DFT(xx) # Señal sin ventana
X_ft = mi_funcion_DFT(xx * v_ft) # Ventana flattop
X_bh = mi_funcion_DFT(xx * v_bh) # Ventana Blackman-Harris
X_h = mi_funcion_DFT(xx * v_h) # Ventana Hamming

f = np.arange(0, N) * fs / N 

SNR_medido= 10*np.log10(np.var(xx)/np.var(na)) #Uso la varianza de xx y na porque son las señales de potencia y ruido reales, Ps y Pn son algebraicas

plt.figure()
plt.plot(f,X,color = 'hotpink')
plt.title('Sin ventana')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X[k]| [db]')    
 
plt.figure()
plt.plot(f,X_ft, color = 'darkturquoise')
plt.title('Ventana Flattop')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X[k]| [db]')       

plt.figure()
plt.plot(f,X_bh, color = 'darkblue')
plt.title('Ventana Blackman-Harris')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X[k]| [db]')    

plt.figure()
plt.plot(f,X_h, color = 'darkmagenta')
plt.title('Ventana Hamming')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X[k]| [db]')    

plt.show()
