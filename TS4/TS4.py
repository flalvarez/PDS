#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 08:36:44 2025

@author: alumno
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


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
plt.plot(ecg_one_lead, color='darkorange')
plt.title('ECG sin ruido')
plt.xlabel('Tiempo [s]') 
plt.ylabel('Amplitud [mV]') 

fw, pw= sig.welch(ecg_one_lead, fs= fs_ecg, window='flattop', nperseg=N/10) #Welch te devuelve la densidad espectral de potencia
# fw2, pw2= sig.welch(ecg_one_lead, fs= fs_ecg, window='flattop', nperseg=N/4) #Welch te devuelve la densidad espectral de potencia
# fw3, pw3= sig.welch(ecg_one_lead, fs= fs_ecg, window='flattop', nperseg=N/40) #Welch te devuelve la densidad espectral de potencia

plt.figure() 
plt.plot(fw,pw, color='darkorange') 
#plt.plot(fw, 20*np.log10(np.abs(pw)), color='darkorange')  #Para ver lo de lo 40db que caen
plt.xlim(0, 40) # Limita el eje x
plt.title('ECG sin ruido - Welch') 
plt.xlabel('Frecuencia [Hz]') 
plt.ylabel('Densidad Espectral de Potencia [V^2 /Hz]') 

# plt.figure() 
# plt.plot(fw2,pw2, color='darkorange') 
# #plt.plot(fw, 20*np.log10(np.abs(pw)), color='darkorange')  #Para ver lo de lo 40db que caen
# plt.xlim(0, 40) # Limita el eje x
# plt.title('ECG sin ruido - Welch - nperseg = N/4) 
# plt.xlabel('Frecuencia [Hz]') 
# plt.ylabel('Densidad Espectral de Potencia [V^2 /Hz]') 

# plt.figure() 
# plt.plot(fw3,pw3, color='darkorange') 
# #plt.plot(fw, 20*np.log10(np.abs(pw)), color='darkorange')  #Para ver lo de lo 40db que caen
# plt.xlim(0, 40) # Limita el eje x
# plt.title('ECG sin ruido - Welch- nperseg = N/40') 
# plt.xlabel('Frecuencia [Hz]') 
# plt.ylabel('Densidad Espectral de Potencia [V^2 /Hz]') 


## En clase hablamos de que el Ancho de banda se mide donde la señal cae 40db
## El ancho de banda concentra la mayor parte de la energia 

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

# plt.figure()
# plt.plot(hb_1, color='mediumaquamarine')
# plt.title('Patron 1')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud [mV]')

# plt.figure()
# plt.plot(hb_2, color='mediumaquamarine')
# plt.title('Patron 2')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud [mV]')

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

## En clase hablamos de que el Ancho de banda se mide donde la señal cae 40db
## El ancho de banda concentra la mayor parte de la energia 


# #%%

# ####################################
# # Lectura de pletismografía (PPG)  #
# ####################################

fs_ppg = 400 # Hz


# ##################
# ## PPG sin ruido
# ##################

ppg = np.load('ppg_sin_ruido.npy')
N = len(ppg)

# # Señal en el tiempo
plt.figure()
t_ppg = np.arange(N) / fs_ppg
plt.plot(t_ppg, ppg, color='indianred')
plt.title('PPG sin ruido')
plt.xlabel('Tiempo [s]') 
plt.ylabel('Amplitud [mV]')


# # Welch
fw_ppg, pw_ppg = sig.welch(ppg, fs=fs_ppg, window='flattop', nperseg=N/10)

plt.figure()
plt.plot(fw_ppg, pw_ppg, color='indianred')
plt.xlim(0, 40)
plt.title('PPG sin ruido - Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad Espectral de Potencia [dB (V²/Hz)]')


pot_acum_ppg = np.cumsum(pw_ppg) * (fw_ppg[1] - fw_ppg[0])
pot_total_ppg = pot_acum_ppg[-1]

idx_ppg = np.where(pot_acum_ppg >= 0.99 * pot_total_ppg)[0][0]
bw_ppg = fw_ppg[idx_ppg]

print("Ancho de banda del PPG sin ruido (99% potencia):", bw_ppg, "Hz")

# ##################
# ## PPG con ruido
# ##################

# # Cargar el archivo CSV como un array de NumPy
ppgR = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe
N = len(ppgR)

# # Señal en el tiempo
plt.figure()
t_ppgR = np.arange(N) / fs_ppg
plt.plot(t_ppgR, ppgR, color='mediumslateblue')
plt.title('PPG con ruido')
plt.xlabel('Tiempo [s]') 
plt.ylabel('Amplitud [mV]')


# # # Welch
fw_ppgR, pw_ppgR = sig.welch(ppgR, fs=fs_ppg, window='flattop', nperseg=N/10)

plt.figure()
plt.plot(fw_ppgR, pw_ppgR, color='mediumslateblue')
plt.xlim(0, 40)
plt.title('PPG con ruido - Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad Espectral de Potencia [dB (V²/Hz)]')


pot_acum_ppgR = np.cumsum(pw_ppgR) * (fw_ppgR[1] - fw_ppgR[0])
pot_total_ppgR = pot_acum_ppgR[-1]

idx_ppgR = np.where(pot_acum_ppgR >= 0.99 * pot_total_ppgR)[0][0]
bw_ppgR = fw_ppgR[idx_ppgR]

print("Ancho de banda del PPG con ruido (99% potencia):", bw_ppgR   , "Hz")


# # #%%

# # ####################
# # # Lectura de audio #
# # ####################

# # # Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# # fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# # fs_audio, wav_data = sio.wavfile.read('silbido.wav')

N = len(wav_data)
t_audio = np.arange(N) / fs_audio


plt.figure()
plt.plot(wav_data, color = 'silver')
plt.title('Audio en el dominio del tiempo')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')

fw_audio, pw_audio = sig.welch(wav_data, fs=fs_audio, window='flattop', nperseg=N/4)

plt.figure()
plt.plot(fw_audio,pw_audio, color='silver')
plt.title('Audio - Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia (V²/Hz)')
plt.xlim(0, 5000)
#plt.xlim(0, fs_audio/2)

pot_acum_audio = np.cumsum(pw_audio) * (fw_audio[1] - fw_audio[0])
pot_total_audio = pot_acum_audio[-1]
idx_audio = np.where(pot_acum_audio >= 0.99 * pot_total_audio)[0][0]
bw_audio = fw_audio[idx_audio]

print("Ancho de banda del audio (99% potencia):", bw_audio, "Hz")

# # --- Blackman-Tukey ---
M = N // 10   # recorte de autocorrelación 
rxx = np.correlate(wav_data, wav_data, mode='full') / N
rxx = rxx[N-1 : N+M]   # quedarse con los retardos positivos

# # aplicar ventana
ventana_bt = np.blackman(len(rxx))
rxx_win = rxx * ventana_bt

# # transformada de Fourier de la autocorrelación
Pxx_bt = np.real(np.fft.fft(rxx_win, n=4096))
f_bt = np.fft.fftfreq(4096, 1/fs_audio)
Pxx_bt = Pxx_bt[:len(f_bt)//2]
f_bt = f_bt[:len(f_bt)//2]

# # --- Gráfico ---
plt.figure()
plt.plot(f_bt, Pxx_bt)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia [V²/Hz]')
plt.title('PSD por método Blackman–Tukey')
plt.xlim(0, 5000)
plt.grid(True)
plt.show()

# # # si quieren oirlo, tienen que tener el siguiente módulo instalado
# # # pip install sounddevice
# # # import sounddevice as sd
# # # sd.play(wav_data, fs_audio)