import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

sampleRate = 44100
time = 5

t = np.linspace(0, time, sampleRate * time)  #  Produces a 5 second Audio-File

signals_wav=[]  #list of synchronized signals

signals_wav.append(np.sin(115 * 2 * np.pi * t)) #First Sine_wave with frequency 60 Hz
signals_wav.append(np.sin(240 * 2 * np.pi * t)) 
signals_wav.append(np.sin(455 * 2 * np.pi * t)) 
signals_wav.append(np.sin(800 * 2 * np.pi * t)) 
signals_wav.append(np.sin(2000 * 2 * np.pi * t)) 
signals_wav.append(np.sin(4500 * 2 * np.pi * t)) 
signals_wav.append(np.sin(9000 * 2 * np.pi * t)) 
signals_wav.append(np.sin(13000 * 2 * np.pi * t)) 
signals_wav.append(np.sin(15000 * 2 * np.pi * t)) 
signals_wav.append(np.sin(20000 * 2 * np.pi * t)) 

for i in signals_wav:
    finalWave=i+1

wavfile.write('Sin.wav', sampleRate, finalWave)

print(len(signals_wav)) #print no of lists-->10
print(len(signals_wav[0])) #print no of samples in each list