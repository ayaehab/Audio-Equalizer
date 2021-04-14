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
        self.ui = uic.loadUi('ui.ui', self)

        self.actionOpen_2.triggered.connect(lambda: self.open_file())
        self.actionOpen.triggered.connect(self.new_window)

        self.Play_Button.clicked.connect(lambda: self.play())

        self.Stop_Button.clicked.connect(lambda: self.stop())

        # Signals Menu

        self.Channel1.triggered.connect(
            lambda checked: (self.select_channel(1)))
        self.Channel2.triggered.connect(
            lambda checked: (self.select_channel(2)))

        self.sliderList = [self.Slider_1, self.Slider_2, self.Slider_3, self.Slider_4, self.Slider_5,
                           self.Slider_6, self.Slider_7, self.Slider_8, self.Slider_9, self.Slider_10]

        self.sliderList[0].valueChanged.connect(lambda: self.equalizer())
        self.sliderList[1].valueChanged.connect(lambda: self.equalizer())
        self.sliderList[2].valueChanged.connect(lambda: self.equalizer())
        self.sliderList[3].valueChanged.connect(lambda: self.equalizer())
        self.sliderList[4].valueChanged.connect(lambda: self.equalizer())
        self.sliderList[5].valueChanged.connect(lambda: self.equalizer())
        self.sliderList[6].valueChanged.connect(lambda: self.equalizer())
        self.sliderList[7].valueChanged.connect(lambda: self.equalizer())
        self.sliderList[8].valueChanged.connect(lambda: self.equalizer())
        self.sliderList[9].valueChanged.connect(lambda: self.equalizer())

        self.mbands = []
        self.fbands = []
        self.gain = [self.sliderList[0].value(), self.sliderList[1].value(), self.sliderList[2].value(),
                     self.sliderList[3].value(), self.sliderList[4].value(
        ), self.sliderList[5].value(),
            self.sliderList[6].value(), self.sliderList[7].value(), self.sliderList[8].value(), self.sliderList[9].value()]


    def new_window(self):
        self.new_win = AudioEqualizer()
        self.new_win.show()
    
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
            self.freq = fftfreq(self.length, 1 / self.samplerate)

            # Plot first channel's signal
            self.InputSignal.plot(self.time, self.data[:, 0], pen=self.pens[0])
            # # Plot second channel's signal
            self.InputSignal.plot(self.time, self.data[:, 1], pen=self.pens[1])
            self.InputSignal.setLimits(xMin=0, xMax=500000, yMin=-200000, yMax=200000)


        else:
            QMessageBox.warning(self.centralWidget,
                                'you must select .wav file')

    def get_extention(self, s):
        for i in range(1, len(s)):
            if s[-i] == '.':
                return s[-(i - 1):]

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

    def equalizer(self):
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

        for index in range(10):
            # we changed it to np.array so we can multiply the value by value not multipling the list that will generate repetation of value not multplication
            Magnified_Magnitued = self.gain[index] * (self.mbands[index])
            self.newMagnitude.append(Magnified_Magnitued)

        for band in self.newMagnitude:
            for magnitude in band:
                self.outputSignal.append(magnitude)

        finalSignal = np.multiply(self.fftPhase, self.outputSignal)
        self.inverse = np.fft.irfft(finalSignal)
        self.OutputSignal.setYRange(min(self.inverse), max(self.inverse))
        self.OutputSignal.plot(self.inverse, pen=pg.mkPen('y'))

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
        if self.Channel2.isChecked():
            sd.play(self.inverse)

    def stop(self):
        if self.Channel1.isChecked():
            sd.stop()
        if self.Channel2.isChecked():
            sd.stop()


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = AudioEqualizer()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
