from PyQt5 import QtCore, QtWidgets, QtGui, uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QMessageBox

import queue
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


import sys
import numpy as np
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

            # The duration is equal to the number of frames divided by the framerate (frames per second)
            self.duration = self.data.shape[0] / self.samplerate

            # Return evenly spaced numbers over a specified interval
            self.time = np.linspace(0., self.duration, self.data.shape[0])

            # Plot first channel's signal
            self.InputSignal.plot(self.time, self.data[:, 0], pen=self.pens[0])
            # Plot second channel's signal
            self.InputSignal.plot(self.time, self.data[:, 1], pen=self.pens[1])

        else:
            QMessageBox.warning(self.centralWidget,
                                'you must select .wav file')

    def get_extention(self, s):
        for i in range(1, len(s)):
            if s[-i] == '.':
                return s[-(i - 1):]


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = AudioEqualizer()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
