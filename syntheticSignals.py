import numpy as np
from scipy.io import wavfile
import soundfile
import matplotlib.pyplot as plt


sampleRate = 22050
time = 1

t = np.linspace(0, time, sampleRate * time)  #  Produces a 1 second Audio-File
finalWave = np.sin(500 * 2 * np.pi * t) + np.sin(1000 * 2 * np.pi * t) + np.sin(1500 * 2 * np.pi * t) + np.sin(2000 * 2 * np.pi * t) + np.sin(2500 * 2 * np.pi * t) + np.sin(3000 * 2 * np.pi * t) + np.sin(3500 * 2 * np.pi * t) + np.sin(4000 * 2 * np.pi * t) + np.sin(4500 * 2 * np.pi * t) + np.sin(5000 * 2 * np.pi * t) 

soundfile.write('Sin.wav', finalWave, sampleRate)
#wavfile.write('Sin.wav', sampleRate, finalWave)

'''
sampleRate = 48000
time = 1

t = np.linspace(0, time, sampleRate * time)  #  Produces a 1 second Audio-File

signals_wav=[]  #list of synchronized signals

signals_wav.append(np.sin(500 * 2 * np.pi * t)) #First Sine_wave with frequency 60 Hz
signals_wav.append(np.sin(1000 * 2 * np.pi * t)) 
signals_wav.append(np.sin(1500 * 2 * np.pi * t)) 
signals_wav.append(np.sin(2000 * 2 * np.pi * t)) 
signals_wav.append(np.sin(2500 * 2 * np.pi * t)) 
signals_wav.append(np.sin(3000 * 2 * np.pi * t)) 
signals_wav.append(np.sin(3500 * 2 * np.pi * t)) 
signals_wav.append(np.sin(4000 * 2 * np.pi * t)) 
signals_wav.append(np.sin(4500 * 2 * np.pi * t)) 
signals_wav.append(np.sin(5000 * 2 * np.pi * t)) 

for i in signals_wav:
    finalWave=i+1

wavfile.write('Sin.wav', sampleRate, finalWave)

print(len(signals_wav)) #print no of lists-->10
print(len(signals_wav[0])) #print no of samples in each list



import numpy as np
import matplotlib.pyplot as plt
import wave
import soundfile
sampling_rate = 48000
amplitude = 16000

t= np.arange(0,1,1/sampling_rate)
sine_wave = (np.sin(2 * np.pi * 500* t)+np.sin(2 * np.pi * 1000* t)+np.sin(2 * np.pi * 1500* t)+np.sin(2 * np.pi * 2000* t)+np.sin(2 * np.pi * 2500* t)+np.sin(2 * np.pi * 3000* t)+np.sin(2 * np.pi * 3500* t))
#wavio.write("sine.wav", x, samplerate, sampwidth=1)

soundfile.write('sinwave.wav', sine_wave, sampling_rate)'''