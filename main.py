import sounddevice as sd
from librosa import display
import librosa
import scipy.io
from scipy.io import wavfile
import pyqtgraph.exporters
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
from scipy import signal
import os
import pyqtgraph as pg
from pyqtgraph import PlotWidget, PlotItem
import pandas as pd
from scipy.fft import fftfreq
from numpy.fft import fft, ifft
import numpy as np
import sys
from PyQt5 import QtCore, QtWidgets, QtGui, uic
from PyQt5.QtCore import pyqtSlot, QSettings
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QMessageBox
import matplotlib
import queue
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas,  NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pdf import GeneratePDF
import img_rc  # for gui

matplotlib.use('Qt5Agg')


class AudioEqualizer(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi('GUI.ui', self)

        self.settings = QSettings("Audio Equalizer", 'App')
        # last_file_opened = self.settings.value("last_file", self.selected_file[0]).toString()
        # self.settings.setValue("last_file", QtCore.QVariant(QtCore.QString('file_name')))
        try:
            # Saving the last position and size of the application
            self.resize(self.settings.value('window size'))
            self.move(self.settings.value('window position'))
        except:
            pass
        # Connecting Buttons
        self.action_open.triggered.connect(lambda: self.browse_file())
        self.action_new_win.triggered.connect(self.make_new_window)
        self.actionSave_as_PDF.triggered.connect(lambda: self.create_my_pdf())
        self.action_clear.triggered.connect(lambda: self.clear_all())

        self.actionPalette_1.triggered.connect(lambda: self.palette_btn1())
        self.actionPalette_2.triggered.connect(lambda: self.palette_btn2())
        self.actionPalette_3.triggered.connect(lambda: self.palette_btn3())
        self.actionPalette_4.triggered.connect(lambda: self.palette_btn4())
        self.actionPalette_5.triggered.connect(lambda: self.palette_btn5())

        self.InputCh.triggered.connect(lambda: self.select_channel(1))
        self.ISpectroCh.triggered.connect(lambda: self.select_channel(2))
        self.OutputCh.triggered.connect(lambda: self.select_channel(3))
        self.OSpectroCh.triggered.connect(lambda: self.select_channel(4))

        self.right_button.clicked.connect(lambda: self.Scroll_right())
        self.left_button.clicked.connect(lambda: self.Scroll_left())
        self.up_button.clicked.connect(lambda: self.Scroll_up())
        self.down_button.clicked.connect(lambda: self.Scroll_down())

        self.zoom_in.clicked.connect(lambda: self.zoomin())
        self.zoom_out.clicked.connect(lambda: self.zoomout())

        self.Play_Button.clicked.connect(lambda: self.play())
        self.Stop_Button.clicked.connect(lambda: self.stop())

        self.pens = [pg.mkPen('r'), pg.mkPen('b'), pg.mkPen('g')]
        self.default_color = {'ticks': [(0.5, (0, 182, 188, 255)),
                                        (1.0, (246, 111, 0, 255)),
                                        (0.0, (75, 0, 113, 255))]}

        self.sliderList = [self.Slider_1, self.Slider_2, self.Slider_3, self.Slider_4, self.Slider_5,
                           self.Slider_6, self.Slider_7, self.Slider_8, self.Slider_9, self.Slider_10]
        self.inverse = np.empty(shape=[0, 1])
        self.Slider_11.valueChanged.connect(
            lambda: self.spec_range(self.inverse))  # MinSlider
        self.Slider_12.valueChanged.connect(
            lambda: self.spec_range(self.inverse))  # MaxSlider

        for i in range(10):
            self.sliderList[i].valueChanged.connect(lambda: self.equalizer())

        self.mbands = []
        self.fbands = []
        self.gain = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def closeEvent(self, event):
        self.settings.setValue('window size', self.size())
        self.settings.setValue('window position', self.pos())

    def make_new_window(self):
        self.new_win = AudioEqualizer()
        self.new_win.show()

    def plot_spectrogram(self, data_col, viewer, color, fs=10e3):
        # im not sure how to compute fs, default value for this task will be 10e3
        fs = 10e3
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
        # Include the units automatimy_pdfy scales the axis and adjusts the SI prefix (in this case kHz)
        viewer.setLabel('left', "Frequency", units='Hz')

        ################# Coloring Spectrogram ############

        # You can adjust it and then save it using hist.gradient.saveState()
        hist.gradient.restoreState({'mode': 'rgb',
                                    'ticks': color['ticks']})
        hist.gradient.showTicks(False)
        hist.shape
        hist.layout.setContentsMargins(0, 0, 0, 0)
        hist.vb.setMouseEnabled(x=False, y=False)

    def palette_btn1(self):  # RGB Grey White Color Palette
        color = {'ticks': [(0.5, (255, 0, 0, 255)),
                           (1.0, (0, 0, 255, 255)),
                           (0.0, (0, 0, 0, 255))]}
        if self.ISpectroCh.isChecked():
            self.plot_spectrogram(
                self.data, self.InputSpectro, color)
        if self.OSpectroCh.isChecked():
            self.plot_spectrogram(
                self.inverse, self.OutputSpectro, color)

    def palette_btn2(self):  # Craftsman Connection Color Palette
        color = {'ticks': [(0.5, (234, 214, 28, 255)),
                           (1.0, (215, 199, 151, 255)),
                           (0.0, (0, 0, 0, 255))]}
        if self.ISpectroCh.isChecked():
            self.plot_spectrogram(
                self.data, self.InputSpectro, color)
        if self.OSpectroCh.isChecked():
            self.plot_spectrogram(
                self.inverse, self.OutputSpectro, color)

    def palette_btn3(self):  # hudcolor Color Palette
        color = {'ticks': [(0.5, (102, 204, 255, 255)),
                           (1.0, (255, 102, 204, 255)),
                           (0.0, (0, 0, 0, 255))]}
        if self.ISpectroCh.isChecked():
            self.plot_spectrogram(
                self.data, self.InputSpectro, color)
        if self.OSpectroCh.isChecked():
            self.plot_spectrogram(
                self.inverse, self.OutputSpectro, color)

    def palette_btn4(self):  # Peacocks & Butterflies Color Palette
        color = {'ticks': [(0.5, (0, 149, 182, 255)),
                           (1.0, (198, 195, 134, 255)),
                           (0.0, (0, 0, 0, 255))]}
        if self.ISpectroCh.isChecked():
            self.plot_spectrogram(
                self.data, self.InputSpectro, color)
        if self.OSpectroCh.isChecked():
            self.plot_spectrogram(
                self.inverse, self.OutputSpectro, color)

    def palette_btn5(self):  # Loznice Color Palette
        color = {'ticks': [(0.5, (242, 226, 205, 255)),
                           (1.0, (166, 158, 176, 255)),
                           (0.0, (0, 0, 0, 255))]}
        if self.ISpectroCh.isChecked():
            self.plot_spectrogram(
                self.data, self.InputSpectro, color)
        if self.OSpectroCh.isChecked():
            self.plot_spectrogram(
                self.inverse, self.OutputSpectro, color)

    def select_channel(self, signal):
        if signal == 1:
            if self.InputCh.isChecked():
                self.InputCh.setChecked(True)
            else:
                self.InputCh.setChecked(False)

        elif signal == 2:
            if self.ISpectroCh.isChecked():
                self.ISpectroCh.setChecked(True)
            else:
                self.ISpectroCh.setChecked(False)

        elif signal == 3:
            if self.OutputCh.isChecked():
                self.OutputCh.setChecked(True)
            else:
                self.OutputCh.setChecked(False)

        elif signal == 4:
            if self.OSpectroCh.isChecked():
                self.OSpectroCh.setChecked(True)
            else:
                self.OSpectroCh.setChecked(False)

# Open (.wav ) file, read it using Scipy Lib, and plot it in inputSignal Viewer

    def browse_file(self):
        self.selected_file = QtGui.QFileDialog.getOpenFileName(
            self, 'Select .wav file ', './', "Raw Data(*.wav)",  os.getenv('HOME'))

        path = str(self.selected_file[0])
        # self.settings.setValue(
        #     "last_file", QtCore.QVariant(QtCore.QString('file_name')))
        self.file_ext = self.get_extention(path)
        # check the file extension is (.Wav)
        if self.file_ext == 'wav':
            # Read selected wav file
            self.samplerate, self.data = wavfile.read(path)
            np.array(self.data)
            self.length = self.data.shape[0]  # number of samples
            # The duration is equal to the number of frames divided by the framerate (frames per second)
            self.duration = (self.length / self.samplerate)

            # Return evenly spaced numbers over a specified interval
            self.time = np.linspace(0., self.duration, self.length)

            self.freq = fftfreq(self.length, 1 / self.samplerate)

            if np.ndim(self.data) == 2:
                # Plot first channel's signal
                self.InputSignal.plot(
                    self.time, self.data[:, 0], pen=self.pens[0])
                self.OutputSignal.plot(
                    self.time, self.data[:, 0], pen=self.pens[0])
                # # Plot second channel's signal on the first one with different color
                self.InputSignal.plot(
                    self.time, self.data[:, 1], pen=self.pens[1])
                self.OutputSignal.plot(
                    self.time, self.data[:, 1], pen=self.pens[1])
                # Plot the spectrogram for the input in InputSpectro Viewer
                self.plot_spectrogram(
                    self.data[:, 0], self.InputSpectro, self.default_color)
            elif np.ndim(self.data) == 1:
                self.InputSignal.plot(self.time, self.data, pen=pg.mkPen('r'))
                self.OutputSignal.plot(self.time, self.data, pen=pg.mkPen('y'))
                self.InputSignal.setYRange(10, -10)
                self.plot_spectrogram(
                    self.data, self.InputSpectro, self.default_color)
            else:
                QMessageBox.warning(self.centralWidget,
                                    "Your .wav file cannot be plotted")

            self.InputSignal.setLimits(
                xMin=0, xMax=500000, yMin=-200000, yMax=200000)

            arr = np.arange(1, (self.samplerate/2)+1, 1)
            self.array = arr[::-1]
            logal = 20 * (np.log10(self.array/(self.samplerate/2))*(-1))

            self.Slider_11.setMinimum(int(logal[0]))
            self.Slider_11.setMaximum(int(logal[-1]))
            self.Slider_12.setMinimum(int(logal[0]))
            self.Slider_12.setMaximum(int(logal[-1]))

            self.Slider_12.setValue(int(logal[0]))
            self.Slider_11.setValue(int(logal[0]))

            self.Slider_11.setSingleStep(int(logal[-1]/10))
            self.Slider_12.setSingleStep(int(logal[-1]/10))

        else:
            QMessageBox.warning(self.centralWidget,
                                'you must select .wav file')

    # Generating the PDF

    # this funcition for the button, when pressed 4 images exported, pdf generated, then the pictures get deleted

    def create_my_pdf(self):
        if self.InputSignal.scene():
            # export all items in all viewers as images
            exporter1 = pg.exporters.ImageExporter(self.InputSignal.scene())
            exporter1.export('input_signal.png')

            exporter3 = pg.exporters.ImageExporter(self.OutputSignal.scene())
            exporter3.export('output_signal.png')

            exporter2 = pg.exporters.ImageExporter(self.InputSpectro.scene())
            exporter2.export('input_spectro.png')

            exporter4 = pg.exporters.ImageExporter(self.OutputSpectro.scene())
            exporter4.export('output_spectro.png')

            my_pdf = GeneratePDF()
            my_pdf.create_pdf()
            my_pdf.save_pdf()

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


#*****************************************END OF Fourrier*************************************#


#*********************************************Equalizer***************************************#

    def equalizer(self):
        self.OutputSignal.clear()

        self.OutputSpectro.clear()

        self.fMagnitude = self.get_fft()[1]

        self.mvaluePerBand = int(len(self.fMagnitude)/10)

        self.fvaluePerBand = int(len(self.freq)/10)

        self.newMagnitude = []

        self.outputSignal = []

        for i in range(10):
            self.mbands.append(
                self.fMagnitude[i * self.mvaluePerBand: min(len(self.fMagnitude)+1, (i+1)*self.mvaluePerBand)])
            self.fbands.append(
                self.freq[i * self.fvaluePerBand: min(len(self.freq)+1, (i+1)*self.fvaluePerBand)])

        for i in range(10):
            self.gain[i] = self.sliderList[i].value()

        for index in range(10):
            # we changed it to np.array so we can multiply the value by value not multipling the list that will generate repetation of value not multplication
            Magnified_Magnitued = np.multiply(
                self.gain[index], (self.mbands[index]))
            self.newMagnitude.append(Magnified_Magnitued)

        for band in self.newMagnitude:
            for magnitude in band:
                self.outputSignal.append(magnitude)
        #get_fft()[2] == fftPhase
        self.inverse = np.fft.irfft(self.outputSignal, len(self.fMagnitude))
        self.OutputSignal.setYRange(np.min(self.inverse), np.max(self.inverse))
        self.OutputSignal.plot(self.time, self.inverse, pen=pg.mkPen('y'))
        self.plot_spectrogram(
            self.inverse, self.OutputSpectro, self.default_color)
        


#*******************************************End of Equalizer**************************************#


    def spec_range(self, data_col):
        if self.Slider_12.value() > self.Slider_11.value():
            fs = self.samplerate
            plt.specgram(data_col, Fs=fs, vmin=self.Slider_11.value(),
                         vmax=self.Slider_12.value())
            # A plot area (ViewBox + axes) for displaying the image
            plt.colorbar()
            plt.show()

            '''Another method to plot Min,Max using matplotlib lib
            self.OutputSpectro.clear()
            self.plot_spectrogram(
                data_col, self.OutputSpectro, self.default_color, fs=fs) '''

#**********************************************toolbar********************************************#

    def play(self):

        if self.InputCh.isChecked():
            sd.play(self.data, self.samplerate)

        elif self.OutputCh.isChecked():
            sd.play(self.inverse)

    def stop(self):

        if self.InputCh.isChecked():
            sd.stop()

        elif self.OutputCh.isChecked():
            sd.stop()


    def clear_all(self):
        self.InputSignal.clear()
        self.InputSpectro.clear()
        self.OutputSignal.clear()
        self.OutputSpectro.clear()
        
    
    def zoomin(self):

        if self.InputCh.isChecked():
            self.InputSignal.plotItem.getViewBox().scaleBy((0.5, 0.5))

        if self.ISpectroCh.isChecked():
            self.InputSpectro.plotItem.getViewBox().scaleBy((0.5, 0.5))

        if self.OutputCh.isChecked():
            self.OutputSignal.plotItem.getViewBox().scaleBy((0.5, 0.5))

        if self.OSpectroCh.isChecked():
            self.OutputSpectro.plotItem.getViewBox().scaleBy((0.5, 0.5))

    def zoomout(self):
        if self.InputCh.isChecked():
            self.InputSignal.plotItem.getViewBox().scaleBy((1.5, 1.5))

        if self.ISpectroCh.isChecked():
            self.InputSpectro.plotItem.getViewBox().scaleBy((1.5, 1.5))

        if self.OutputCh.isChecked():
            self.OutputSignal.plotItem.getViewBox().scaleBy((1.5, 1.5))

        if self.OSpectroCh.isChecked():
            self.OutputSpectro.plotItem.getViewBox().scaleBy((1.5, 1.5))

    def Scroll_right(self):

        if self.InputCh.isChecked():
            self.range = self.InputSignal.getViewBox().viewRange()
            if self.range[0][1] < max(self.time):
                self.InputSignal.getViewBox().translateBy(x=+0.2, y=0)

        if self.OutputCh.isChecked():
            self.range = self.OutputSignal.getViewBox().viewRange()
            if self.range[0][1] < max(self.time):
                self.OutputSignal.getViewBox().translateBy(x=+0.2, y=0)

    def Scroll_left(self):

        if self.InputCh.isChecked():
            self.range = self.InputSignal.getViewBox().viewRange()
            if self.range[0][0] > min(self.time):
                self.InputSignal.getViewBox().translateBy(x=-0.2, y=0)

        if self.OutputCh.isChecked():
            self.range = self.OutputSignal.getViewBox().viewRange()
            if self.range[0][0] > min(self.time):
                self.OutputSignal.getViewBox().translateBy(x=-0.2, y=0)

    def Scroll_up(self):

        if self.InputCh.isChecked():
            self.range = self.InputSignal.getViewBox().viewRange()
            if self.range[1][1] < max(self.data):
                self.InputSignal.getViewBox().translateBy(x=0, y=+0.2)

        if self.OutputCh.isChecked():
            self.range = self.OutputSignal.getViewBox().viewRange()
            if self.range[1][1] < max(self.data):
                self.OutputSignal.getViewBox().translateBy(x=0, y=+0.2)

    def Scroll_down(self):

        if self.InputCh.isChecked():
            self.range = self.InputSignal.getViewBox().viewRange()
            if self.range[1][0] > min(self.data):
                self.InputSignal.getViewBox().translateBy(x=0, y=-0.2)

        if self.ISpectroCh.isChecked():
            self.range = self.InputSpectro.getViewBox().viewRange()
            if self.range[1][0] > min(self.inverse):
                self.InputSpectro.getViewBox().translateBy(x=0, y=-0.2)

        if self.OutputCh.isChecked():
            self.range = self.OutputSignal.getViewBox().viewRange()
            if self.range[1][0] > min(self.data):
                self.OutputSignal.getViewBox().translateBy(x=0, y=-0.2)

        if self.OSpectroCh.isChecked():
            self.range = self.OutputSpectro.getViewBox().viewRange()
            if self.range[1][0] > min(self.data):
                self.OutputSpectro.getViewBox().translateBy(x=0, y=-0.2)

    def clear_all(self):
        self.InputSignal.clear()
        self.InputSpectro.clear()
        self.OutputSignal.clear()
        self.OutputSpectro.clear()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("CUFE")
    app.setOrganizationDomain("CUFEDomain")
    app.setApplicationName("Audio Equalizer")
    application = AudioEqualizer()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
