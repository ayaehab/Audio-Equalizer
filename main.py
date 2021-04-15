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
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QMessageBox
import matplotlib
import queue
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas,  NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pdf import GeneratePDF
import img_rc #for gui

matplotlib.use('Qt5Agg')



class AudioEqualizer(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi('GUI.ui', self)

        #Connecting Buttons
        self.action_open.triggered.connect(lambda: self.browse_file())
        self.action_new_win.triggered.connect(self.make_new_window)
        self.actionSave_as_PDF.triggered.connect(lambda: self.create_my_pdf())
        self.action_clear.triggered.connect(lambda: self.clear_all())
       
        self.right_button.clicked.connect(lambda: self.Scroll_right())
        self.left_button.clicked.connect(lambda: self.Scroll_left())
        self.up_button.clicked.connect(lambda: self.Scroll_up())
        self.down_button.clicked.connect(lambda: self.Scroll_down())
        
        self.zoom_in.clicked.connect(lambda: self.zoomin())
        self.zoom_out.clicked.connect(lambda: self.zoomout())
        
        self.Play_Button.clicked.connect(lambda: self.play())
        self.Stop_Button.clicked.connect(lambda: self.stop())
        self.pens = [pg.mkPen('r'), pg.mkPen('b'), pg.mkPen('g')]
        
        self.sliderList = [self.Slider_1, self.Slider_2, self.Slider_3, self.Slider_4, self.Slider_5,
                           self.Slider_6, self.Slider_7, self.Slider_8, self.Slider_9, self.Slider_10]

        for i in range(10):
            self.sliderList[i].valueChanged.connect(lambda: self.equalizer())

        self.mbands = []
        self.fbands = []
        self.gain = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def make_new_window(self):
        self.new_win = AudioEqualizer()
        self.new_win.show()


    def plot_spectrogram(self, data_col, viewer):
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
        hist.gradient.restoreState(
            {'mode': 'rgb',
                'ticks': [(0.5, (0, 182, 188, 255)),
                          (1.0, (246, 111, 0, 255)),
                          (0.0, (75, 0, 113, 255))]})
        hist.gradient.showTicks(False)
        hist.shape
        hist.layout.setContentsMargins(0, 0, 0, 0)
        hist.vb.setMouseEnabled(x=False, y=False)


# Open (.wav ) file, read it using Scipy Lib, and plot it in inputSignal Viewer


    def browse_file(self):
        self.selected_file = QtGui.QFileDialog.getOpenFileName(
            self, 'Select .wav file ', './', "Raw Data(*.wav)",  os.getenv('HOME'))

        path = str(self.selected_file[0])
        print(path)
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
                # # Plot second channel's signal on the first one with different color
                self.InputSignal.plot(
                    self.time, self.data[:, 1], pen=self.pens[1])
                # Plot the spectrogram for the input in InputSpectro Viewer
                self.plot_spectrogram(self.data[:, 0], self.InputSpectro)
            elif np.ndim(self.data) == 1:
                self.InputSignal.plot(self.time, self.data, pen=pg.mkPen('r'))
                self.plot_spectrogram(self.data, self.InputSpectro)
            else:
                QMessageBox.warning(self.centralWidget,
                                    "Your .wav file cannot be plotted")

            self.InputSignal.setLimits(
                xMin=0, xMax=500000, yMin=-200000, yMax=200000)

        else:
            QMessageBox.warning(self.centralWidget,
                                'you must select .wav file')

    # Generating the PDF

    # this funcition for the button, when pressed 4 images exported, pdf generated, then the pictures get deleted

    def create_my_pdf(self):
        if self.InputSignal.scene():
            #export all items in all viewers as images 
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

            # delete created images after generating PDF file ##### NOT YET ######

            # if os.path.exists("Audio Equalizer.pdf"):
            #     os.remove("input_signal.png")
            #     os.remove("output_signal.png")
            #     os.remove("input_spectro.png")
            #     os.remove("output_spectro.png")

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

#*******************************************End of Equalizer**************************************#        

    def play(self):
        sd.play(self.data, self.samplerate)

    def stop(self):
        sd.stop()
        
        
#**********************************************toolbar********************************************#

    # Zoomin function connected to Zoomin button based on which channel is controlled
    def zoomin(self):
        if self.Channel1.isChecked():
            self.InputSignal.plotItem.getViewBox().scaleBy((0.5, 0.5))

        if self.Channel2.isChecked():
            self.OutputSignal.plotItem.getViewBox().scaleBy((0.5, 0.5))


    # Zoomout function connected to zoomout button based on which channel is controlled
    def zoomout(self):
        if self.Channel1.isChecked():
            self.InputSignal.plotItem.getViewBox().scaleBy((1.5, 1.5))

        if self.Channel2.isChecked():
            self.OutputSignal.plotItem.getViewBox().scaleBy((1.5, 1.5))

   
    #scrolling function connected to scroll buttons based on which channel is controlled
   
   def Scroll_right(self):
        
        if self.Channel1.isChecked():
            
            self.range = self.InputSignal.getViewBox().viewRange()
            if self.range[0][1] < max(self.time) :
                self.InputSignal.getViewBox().translateBy(x=+1, y=0)

        if self.Channel2.isChecked():
            
            self.range = self.OutputSignal.getViewBox().viewRange()
            if self.range[0][1] < max(self.x2) :
                self.OutputSignal.getViewBox().translateBy(x=+1, y=0)


    def Scroll_left(self):
        
        if self.Channel1.isChecked():
            
            self.range = self.InputSignal.getViewBox().viewRange()
            if self.range[0][0] > min(self.time) :
                self.InputSignal.getViewBox().translateBy(x=-1, y=0)

        if self.Channel2.isChecked():
            
            self.range = self.OutputSignal.getViewBox().viewRange()
            if self.range[0][0] > min(self.time) :
                self.OutputSignal.getViewBox().translateBy(x=-1, y=0)
   

    def Scroll_up(self):
        
        if self.Channel1.isChecked():
          
            self.range = self.InputSignal.getViewBox().viewRange()
            if self.range[1][1] < max(self.data) :
                self.InputSignal.getViewBox().translateBy(x=0, y=+0.2)

        if self.Channel2.isChecked():
            
            self.range = self.OutputSignal.getViewBox().viewRange()
            if self.range[1][1] < max(self.inverse) :
                self.OutputSignal.getViewBox().translateBy(x=0, y=+0.2)


    def Scroll_down(self):
        
        if self.Channel1.isChecked():
            
            self.range = self.InputSignal.getViewBox().viewRange()
            if self.range[1][0] > min(self.data) :
                self.InputSignal.getViewBox().translateBy(x=0, y=-0.2)

        if self.Channel2.isChecked():
            
            self.range = self.OutputSignal.getViewBox().viewRange()
            if self.range[1][0] > min(self.inverse) :
                self.OutputSignal.getViewBox().translateBy(x=0, y=-0.2)


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = AudioEqualizer()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
