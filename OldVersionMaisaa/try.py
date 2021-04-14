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
from scipy import signal
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
import pyqtgraph.exporters
from scipy.io import wavfile
import scipy.io
import librosa
from librosa import display
import sounddevice as sd


import matplotlib
matplotlib.use('Qt5Agg')


class AudioEqualizer(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()


        self.ui = uic.loadUi('mui.ui', self)


        self.actionOpen_2.triggered.connect(lambda: self.open_file())

        self.Play_Button.clicked.connect(lambda: self.play())

        self.Stop_Button.clicked.connect(lambda: self.stop())

        # Signals Menu

        self.Channel1.triggered.connect(
            lambda checked: (self.select_channel(1)))
        self.Channel2.triggered.connect(
            lambda checked: (self.select_channel(2)))

        self.sliderList = [self.Slider_1, self.Slider_2, self.Slider_3, self.Slider_4, self.Slider_5,
                           self.Slider_6, self.Slider_7, self.Slider_8, self.Slider_9, self.Slider_10]

        for i in range(10):
            self.sliderList[i].valueChanged.connect(lambda: self.equalizer())

        self.mbands = []
        self.fbands = []
        self.gain = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

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

            self.length = self.data.shape[0]  # number of samples
            # The duration is equal to the number of frames divided by the framerate (frames per second)
            self.duration = self.length / self.samplerate

            # Return evenly spaced numbers over a specified interval
            self.time = np.linspace(0., self.duration, self.length)

            
            self.InputSignal.plot(self.time, self.data, pen=pg.mkPen('r'))
            self.plot_spectrogram(self.data, self.InputSpectro)
            self.freq = fftfreq(self.length, 1 / self.samplerate)
            self.InputSignal.setLimits(
                xMin=0, xMax=500000, yMin=-200000, yMax=200000)

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

    def make_new_window(self):
        self.new_win = AudioEqualizer()
        self.new_win.show()

    def plot_spectrogram(self, data_col, viewer):
        # im not sure how to compute fs, default value for this task will be 10e3
        fs = self.samplerate
        # make sure the data given in array form
        data = np.array(data_col)

        # f : Array of sample frequencies; t : Array of segment times; Sxx : Spectrogram of x. The last axis of Sxx corresponds to the segment times.
        f, t, Sxx = signal.spectrogram(data, fs)

        # A plot area (ViewBox + axes) for displaying the image
        plot_area = viewer.plot()
        # Item for displaying image data
        img = pg.ImageItem()
        viewer.addItem(img)
        # Add a histogram with which to control the gradient of the image
        hist = pg.HistogramLUTItem()
        # Link the histogram to the image
        hist.setImageItem(img)
        # If you don't add the histogram to the window, it stays invisible, but I find it useful.
        viewer.addItem(plot_area)

        # Show the window
        viewer.show()
        # Fit the min and max levels of the histogram to the data available
        hist.setLevels(np.min(Sxx), np.max(Sxx))

        # Sxx contains the amplitude for each pixel
        img.setImage(Sxx)
        # Scale the X and Y Axis to time and frequency (standard is pixels)
        img.scale(t[-1]/np.size(Sxx, axis=1),
                  f[-1]/np.size(Sxx, axis=0))
        # Limit panning/zooming to the spectrogram
        viewer.setLimits(xMin=0, xMax=t[-1], yMin=0, yMax=f[-1])
        # Add labels to the axis
        viewer.setLabel('bottom', "Time", units='s')
        # Include the units automatically scales the axis and adjusts the SI prefix (in this case kHz)
        viewer.setLabel('left', "Frequency", units='Hz')

        ################# Coloring Spectrogram ############

        # You can adjust it and then save it using hist.gradient.saveState()
        hist.gradient.restoreState(
            {'mode': 'rgb',
                'ticks': [(0.5, (0, 182, 188, 255)),
                          (1.0, (246, 111, 0, 255)),
                          (0.0, (75, 0, 113, 255))]})
        hist.gradient.showTicks(False)
        hist.shape
        hist.layout.setContentsMargins(0, 0, 0, 0)
        hist.vb.setMouseEnabled(x=False, y=False)

#*******************************************Fourrier**************************************#

    def get_fft(self):
        # fft returns an array contains all +ve values then all -ve values
        # it has some real and some complex values
        self.fftArray = fft(self.data)
        # separate the +ve from the -ve
        self.fftArrayPositive = self.fftArray[:self.length // 2]
        self.fftArrayNegative = np.flip(self.fftArray[self.length // 2:])
        # get the magnitude of both +ve & -ve
        self.fftArrayAbs = np.abs(self.fftArray)
        self.fftPhase = np.angle(self.fftArray)
        # magnitude of +ve only
        self.fftMagnitude = self.fftArrayAbs[: self.length // 2]

        # plt.plot(self.freq , self.)
        # plt.show()

        # return fftArray that will be used later in the inverse forrier transform
        # and fftMagnitude that will be used in plotting ....
        return self.fftArray, self.fftArrayAbs, self.fftPhase, self.fftMagnitude


#*******************************************END OF Fourrier**************************************#

    def play(self):
        sd.play(self.inverse)

    def stop(self):
        sd.stop()

    def equalizer(self):
        self.OutputSignal.clear()

        self.OutputSpectro.clear()

        self.fMagnitude = self.get_fft()[1]

        self.mvaluePerBand = int(len(self.fMagnitude)/10)

        self.fvaluePerBand = int(len(self.freq)/10)

        self.newMagnitude = []

        self.outputSignal = []
        # print(self.valuePerBand)

        for i in range(10):
            self.mbands.append(
                self.fMagnitude[i * self.mvaluePerBand: min(len(self.fMagnitude)+1, (i+1)*self.mvaluePerBand)])
            self.fbands.append(
                self.freq[i * self.fvaluePerBand: min(len(self.freq)+1, (i+1)*self.fvaluePerBand)])

        for i in range(10):
            self.gain[i] = self.sliderList[i].value()

        print(self.gain)

        for index in range(10):
            # we changed it to np.array so we can multiply the value by value not multipling the list that will generate repetation of value not multplication
            Magnified_Magnitued = self.gain[index] * (self.mbands[index])
            self.newMagnitude.append(Magnified_Magnitued)

        for band in self.newMagnitude:
            for magnitude in band:
                self.outputSignal.append(magnitude)  
        #get_fft()[2] == fftPhase
        finalSignal = np.multiply(self.get_fft()[2], self.outputSignal) 
        self.inverse = np.fft.irfft(finalSignal)
        self.OutputSignal.setYRange(min(self.inverse), max(self.inverse))
        self.OutputSignal.plot(self.inverse, pen=pg.mkPen('y'))
        self.plot_spectrogram(self.inverse, self.OutputSpectro)



'''
    def select_channel(self, channel):
        if channel == 1:
            self.Channel1.setChecked(True)
            self.Channel2.setChecked(False)

        elif channel == 2:
            self.Channel1.setChecked(False)
            self.Channel2.setChecked(True)

    def play(self):
        if self.Channel1.isChecked():
            sd.play(self.data, self.samplerate)

        elif self.Channel1.isChecked():
            sd.play(self.inverse)
'''


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = AudioEqualizer()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
