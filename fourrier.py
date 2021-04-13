import numpy as np
import librosa 
from librosa import display
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

class FT:
    def __init__(self,samples,samplingRate):
        
        self.samples = samples
        self.samplingRate = samplingRate
        
        self.length = len(self.samples)     #number of samples  
        self.duration = self.length/self.samplingRate   #the duration of the audio
        
        self.freq = np.linspace(0, self.samplingRate / 2, int(self.length / 2)) #generate array of freq values to use it in freq domain
        self.time = np.linspace(0, self.duration, self.length) #generate an array of time values to use it in time domain 
        
        
    def get_fft (self):
        # fft returns an array contains all +ve values then all -ve values
        # it has some real and some complex values
        self.fftArray = fft(self.samples) 
        #separate the +ve from the -ve
        self.fftArrayPositive = self.fftArray[:self.length // 2]
        self.fftArrayNegative = np.flip(self.fftArray[self.length // 2:])
        #get the magnitude of both +ve & -ve
        self.fftArrayAbs = np.abs(self.fftArray)
        self.fftMagnitude = self.fftArrayAbs[: self.length // 2] # magnitude of +ve only
        
        # plt.plot(self.freq , self.fftMagnitude)
        # plt.show()
        
        # return fftArray that will be used later in the inverse forrier transform
        # and fftMagnitude that will be used in plotting ....
        return self.fftArray , self.fftMagnitude
    
    
    def get_ifft(self,fftArray):
        
        self.ifftArray = ifft(self.fftArray).real 
        # plt.plot(self.time , self.ifftArray)
        # plt.show()      
        return self.ifftArray
        
        
            
            

#read the audio file and assign its samples/data into an array and its sampling rate in a variable
samples, samplingRate = librosa.load("Sin.wav", sr= None, mono=True, offset=0.0, duration=None)

#create an instance of FT class that contains the methods we are going to use and pass the data and the sampling rate to it
sig = FT(samples, samplingRate)

forrierArray, fMagnitude= sig.get_fft()
i_forrierArray = sig.get_ifft(forrierArray)
# librosa.display.waveplot(y=samples, sr= samplingRate)
# plt.show()

