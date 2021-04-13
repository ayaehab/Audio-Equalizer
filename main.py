from PyQt5 import QtCore, QtWidgets, QtGui, uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QMessageBox

import queue
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


import sys
import numpy as np
from numpy.fft import fft, ifft
from scipy.fft import fftfreq
import pandas as pd
from pyqtgraph import PlotWidget, PlotItem
import pyqtgraph as pg
import os
# import img_rc
from scipy import signal
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
import pyqtgraph.exporters
from scipy.io import wavfile
import scipy.io
import librosa
from librosa import display

import matplotlib
matplotlib.use('Qt5Agg')


class AudioEqualizer(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi('ui.ui', self)

        self.actionOpen_2.triggered.connect(lambda: self.open_file())
        self.pens = [pg.mkPen('r'), pg.mkPen('b'), pg.mkPen('g')]
        
        self.sliderList = [ self.Slider_1, self.Slider_2, self.Slider_3, self.Slider_4, self.Slider_5,
                              self.Slider_6, self.Slider_7, self.Slider_8, self.Slider_9, self.Slider_10 ]
        self.mbands=[]
        self.fbands = []
        self.gain=[]
        
        
    # Open (.wav ) file, read it using Scipy Lib, and plot it in inputSignal Viewer
    def open_file(self):
        self.selected_file = QtGui.QFileDialog.getOpenFileName(
            self, 'Select .wav file ', os.getenv('HOME'))

        self.file_ext = self.get_extention(self.selected_file[0])
        # check the file extension is (.Wav)
        if self.file_ext == 'wav':
            # Read selected wav file
            self.samplerate, self.data = wavfile.read(
                str(self.selected_file[0]))
            
            self.length = self.data.shape[0] #number of samples
            # The duration is equal to the number of frames divided by the framerate (frames per second)
            self.duration = self.length / self.samplerate

            # Return evenly spaced numbers over a specified interval
            self.time = np.linspace(0., self.duration, self.length)
            self.freq = fftfreq(self.length, 1 / self.samplerate)

            # Plot first channel's signal
            # self.InputSignal.plot(self.time, self.data[:, 0], pen=self.pens[0])
            # # Plot second channel's signal
            # self.InputSignal.plot(self.time, self.data[:, 1], pen=self.pens[1])

        else:
            QMessageBox.warning(self.centralWidget,
                                'you must select .wav file')

    def get_extention(self, s):
        for i in range(1, len(s)):
            if s[-i] == '.':
                return s[-(i - 1):]
            
#*******************************************Fourrier**************************************#  
       
    def get_fft (self):
        # fft returns an array contains all +ve values then all -ve values
        # it has some real and some complex values
        self.fftArray = fft(self.data) 
        #separate the +ve from the -ve
        self.fftArrayPositive = self.fftArray[:self.length // 2]
        self.fftArrayNegative = np.flip(self.fftArray[self.length // 2:])
        #get the magnitude of both +ve & -ve
        self.fftArrayAbs = np.abs(self.fftArray)
        self.fftPhase = np.angle(self.fftArray)
        self.fftMagnitude = self.fftArrayAbs[: self.length // 2] # magnitude of +ve only
        
        # plt.plot(self.freq , self.)
        # plt.show()
        
        # return fftArray that will be used later in the inverse forrier transform
        # and fftMagnitude that will be used in plotting ....
        return self.fftArray ,self.fftArrayAbs , self.fftPhase, self.fftMagnitude
    
    
    def get_ifft(self,fftArray):
        
        self.ifftArray = ifft(self.fftArray).real 
        # plt.plot(self.time , self.ifftArray)
        # plt.show()      
        return self.ifftArray
    
#*******************************************END OF Fourrier**************************************# 
   
    def slider(self):
        
        self.fMagnitude = self.get_fft()[1]
        self.mvaluePerBand = int(len(self.fMagnitude)/10)
        self.fvaluePerBand = int(len(self.freq)/10)
        #print(self.valuePerBand)
        
        for i in range(10):
            self.mbands.append(self.fMagnitude[i * self.mvaluePerBand : min(len(self.fMagnitude)+1, (i+1)*self.mvaluePerBand)])
            self.fbands.append(self.freq[i * self.fvaluePerBand : min(len(self.freq)+1, (i+1)*self.fvaluePerBand)])
            
    def equalizer(self):
        self.newMagnitude = []
        for i in range(10):
            self.gain.append(self.sliderList[i].value())
        
        for index in range(10):
            Magnified_Magnitued = self.gain[i] * np.array(self.bands[i]) #we changed it to np.array so we can multiply the value by value not multipling the list that will generate repetation of value not multplication
            self.newMagnitude.append(list(Magnified_Magnitued))

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = AudioEqualizer()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
