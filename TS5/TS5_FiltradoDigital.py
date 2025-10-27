#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 06:00:27 2025

@author: alumno
"""
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz


##################
# Plantilla #
##################




##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')
N = len(ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead, color='darkorange')
plt.title('ECG sin ruido')
plt.xlabel('Tiempo [s]') 
plt.ylabel('Amplitud [mV]') 

fw, pw= sig.welch(ecg_one_lead, fs= fs_ecg, window='flattop', nperseg=N/10) #Welch te devuelve la densidad espectral de potencia
# fw2, pw2= sig.welch(ecg_one_lead, fs= fs_ecg, window='flattop', nperseg=N/4) #Welch te devuelve la densidad espectral de potencia
# fw3, pw3= sig.welch(ecg_one_lead, fs= fs_ecg, window='flattop', nperseg=N/40) #Welch te devuelve la densidad espectral de potencia

plt.figure() 
#plt.plot(fw,pw, color='darkorange') 
plt.plot(fw, 20*np.log10(np.abs(pw)), color='darkorange')  #Para ver lo de lo 40db que caen
plt.xlim(0, 40) # Limita el eje x
plt.title('ECG sin ruido - Welch') 
plt.xlabel('Frecuencia [Hz]') 
plt.ylabel('Densidad Espectral de Potencia [V^2 /Hz]') 


pot_acum = np.cumsum(pw) * (fw[1]-fw[0])  # Cumsum te hace la integral. Hay que multiplicarlo por un diferencial delta f para ir acumulando
pot_total = pot_acum[-1]                  # Cuando pones el [-1] te devuelve la potencia del ultimo valor y como estoy usando la potencia acumulada seria mi potencia total
 
idx = np.where(pot_acum >= 0.99 * pot_total)[0][0] # Return elements chosen from x or y depending on condition. Te arma un vector que te dice si se cumple o no la condicion Me devuelve el primer valor donde se cumple esta condicion
bw = fw[idx] # Me fijo a que frecuencia corresponde el valor
print("Ancho de banda del ECG sin ruido (99% potencia):", bw, "Hz")


##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecgR_one_lead = mat_struct['ecg_lead']
N = len(ecgR_one_lead)
hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

plt.figure()
plt.plot(ecg_one_lead[5000:12000], color='mediumaquamarine')
plt.title('ECG con ruido')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')

plt.figure()
plt.plot(hb_1, color='mediumaquamarine')
plt.title('Latido normal')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [mV]')

plt.figure()
plt.plot(hb_2, color='mediumaquamarine')
plt.title('Latido ventricular')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [mV]')

# Welch
fw_ecgR, pw_ecgR = sig.welch(ecg_one_lead, fs=fs_ecg, window='flattop', nperseg=N/40)

plt.figure()
# plt.plot(fw_ecgR, pw_ecgR, color='mediumaquamarine')
plt.plot(fw_ecgR, 20*np.log10(np.abs(pw_ecgR)), color='mediumaquamarine')  #Para ver lo de lo 40db que caen
plt.xlim(0, 40)
plt.title('ECG con ruido - Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad Espectral de Potencia [V²/Hz]')

# Potencia acumulada
pot_acum_ecgR = np.cumsum(pw_ecgR) * (fw_ecgR[1] - fw_ecgR[0])
pot_total_ecgR = pot_acum_ecgR[-1]

# Frecuencia donde se acumula el 99% de la potencia
idx_ecgR = np.where(pot_acum_ecgR >= 0.99 * pot_total_ecgR)[0][0]
bw_ecgR = fw_ecgR[idx_ecgR]

print("Ancho de banda del ECG con ruido (99% potencia):", bw_ecgR, "Hz")