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

#%%

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')
N = len(ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead, color='crimson')
plt.title('ECG sin ruido')
plt.xlabel('Tiempo [s]') 
plt.ylabel('Amplitud [mV]') 

fw, pw= sig.welch(ecg_one_lead, fs= fs_ecg, window='flattop', nperseg=N/10) #Welch te devuelve la densidad espectral de potencia

plt.figure() 
plt.plot(fw,pw, color='crimson') 
#plt.plot(fw, 20*np.log10(np.abs(pw)), color='crimson')  #Para ver lo de lo 40db que caen
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
plt.plot(ecg_one_lead[5000:12000], color='pink')
plt.title('ECG con ruido')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')

plt.figure()
plt.plot(hb_1, label='Latido normal', color='yellowgreen')
plt.plot(hb_2, label='Latido ventricular', color='orange')
plt.title('Latidos promedios')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [mV]')
plt.legend()

# Welch
fw_ecgR, pw_ecgR = sig.welch(ecg_one_lead, fs=fs_ecg, window='flattop', nperseg=N/40)

plt.figure()
plt.plot(fw_ecgR, pw_ecgR, color='mediumaquamarine')
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

#############
# Plantilla #
#############

nyq_frec = fs_ecg / 2
ripple = 1      # dB
atenuacion = 40 # dB

ws1 = 0.2
wp1 = 0.6
wp2 = 30
ws2 = 40

frecs = np.array([0.0, ws1, wp1, wp2, ws2, nyq_frec]) / nyq_frec #Normalizo las frecuencias para usarlas en iirdesign
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
gains = 10**(gains/20)

## Plantilla
# plt.figure()
# plt.plot(frecs*nyq_frec, 20*np.log10(gains), color="lightblue")
# plt.title('Plantilla de diseño del filtro ECG')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Magnitud [dB]')
# plt.grid(True)

wp = [wp1 / nyq_frec, wp2 / nyq_frec]
ws = [ws1 / nyq_frec, ws2 / nyq_frec]

# Armo el filtro Butterworth
sos_b = sig.iirdesign(wp, ws, gpass=ripple, gstop=atenuacion, ftype='butter', output='sos')

# Returns:
# w: ndarray
#   The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample).
# h: ndarray
#   The frequency response, as complex numbers.
# wornN controla la resolucion en frecuencia
w, h = sig.sosfreqz(sos_b, worN=4096, fs=fs_ecg)

# --- Gráfico Bode + plantilla ---
plt.figure()
plt.semilogx(w, 20 * np.log10(np.abs(h)), color="black", label='Butterworth (SOS)') #semilog te pone el eje en radianes para poder hacer el bode :)
plt.axhline(-ripple, color='green', linestyle='--', label=f'Límite banda de paso ({ripple} dB)') #axhline es para marcar los limites en db (1db para el ripple y 40db para la atenuacion)
plt.axhline(-atenuacion, color='red', linestyle='--', label=f'Límite banda de stop ({atenuacion} dB)')
# Bandas de la plantilla: axvspan te colorea las bandas de paso y de stop
plt.axvspan(0, ws1, color='red', alpha=0.2, label='Stop baja')
plt.axvspan(wp1, wp2, color='green', alpha=0.2, label='Banda de paso')
plt.axvspan(ws2, fs_ecg / 2.0, color='red', alpha=0.2, label='Stop alta')
plt.title('Filtro Butterworth con plantilla de diseño')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(0.1, fs_ecg / 2)
plt.grid(which='both')
plt.legend()
plt.show()

# Armo el filtro Chebyshev tipo 1
sos_c1 = sig.iirdesign(wp, ws, gpass=ripple, gstop=atenuacion, ftype='cheby1', output='sos')

w, h = sig.sosfreqz(sos_c1, worN=4096, fs=fs_ecg)

# --- Gráfico Bode + plantilla ---
plt.figure()
plt.semilogx(w, 20 * np.log10(np.abs(h)), color="black", label='Chevychev tipo 1 (SOS)') #semilog te pone el eje en radianes para poder hacer el bode :)
plt.axhline(-ripple, color='green', linestyle='--', label=f'Límite banda de paso ({ripple} dB)') #axhline es para marcar los limites en db (1db para el ripple y 40db para la atenuacion)
plt.axhline(-atenuacion, color='red', linestyle='--', label=f'Límite banda de stop ({atenuacion} dB)')

# Bandas de la plantilla: axvspan te colorea las bandas de paso y de stop
plt.axvspan(0, ws1, color='red', alpha=0.2, label='Stop baja')
plt.axvspan(wp1, wp2, color='green', alpha=0.2, label='Banda de paso')
plt.axvspan(ws2, fs_ecg / 2.0, color='red', alpha=0.2, label='Stop alta')

plt.title('Filtro Chebyshev tipo I con plantilla de diseño')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(0.1, fs_ecg / 2)
plt.grid(which='both')
plt.legend()
plt.show()

# Armo el filtro Chebyshev tipo 2
sos_c1 = sig.iirdesign(wp, ws, gpass=ripple, gstop=atenuacion, ftype='cheby2', output='sos')

w, h = sig.sosfreqz(sos_c1, worN=4096, fs=fs_ecg)

# --- Gráfico Bode + plantilla ---
plt.figure()
plt.semilogx(w, 20 * np.log10(np.abs(h)), color="black", label='Chevychev tipo 2 (SOS)') #semilog te pone el eje en radianes para poder hacer el bode :)
plt.axhline(-ripple, color='green', linestyle='--', label=f'Límite banda de paso ({ripple} dB)') #axhline es para marcar los limites en db (1db para el ripple y 40db para la atenuacion)
plt.axhline(-atenuacion, color='red', linestyle='--', label=f'Límite banda de stop ({atenuacion} dB)')

# Bandas de la plantilla: axvspan te colorea las bandas de paso y de stop
plt.axvspan(0, ws1, color='red', alpha=0.2, label='Stop baja')
plt.axvspan(wp1, wp2, color='green', alpha=0.2, label='Banda de paso')
plt.axvspan(ws2, fs_ecg / 2.0, color='red', alpha=0.2, label='Stop alta')

plt.title('Filtro Chebyshev tipo II con plantilla de diseño')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(0.1, fs_ecg / 2)
plt.grid(which='both')
plt.legend()
plt.show()

# Armo el filtro Cauer
sos_c1 = sig.iirdesign(wp, ws, gpass=ripple, gstop=atenuacion, ftype='ellip', output='sos')

w, h = sig.sosfreqz(sos_c1, worN=4096, fs=fs_ecg)

# --- Gráfico Bode + plantilla ---
plt.figure()
plt.semilogx(w, 20 * np.log10(np.abs(h)), color="black", label='Cauer (SOS)') #semilog te pone el eje en radianes para poder hacer el bode :)
plt.axhline(-ripple, color='green', linestyle='--', label=f'Límite banda de paso ({ripple} dB)') #axhline es para marcar los limites en db (1db para el ripple y 40db para la atenuacion)
plt.axhline(-atenuacion, color='red', linestyle='--', label=f'Límite banda de stop ({atenuacion} dB)')

# Bandas de la plantilla: axvspan te colorea las bandas de paso y de stop
plt.axvspan(0, ws1, color='red', alpha=0.2, label='Stop baja')
plt.axvspan(wp1, wp2, color='green', alpha=0.2, label='Banda de paso')
plt.axvspan(ws2, fs_ecg / 2.0, color='red', alpha=0.2, label='Stop alta')

plt.title('Filtro Cauer con plantilla de diseño')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.xlim(0.1, fs_ecg / 2)
plt.grid(which='both')
plt.legend()
plt.show()