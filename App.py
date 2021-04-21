import sounddevice as sd
import scipy.io
from scipy.io import wavfile
import pyqtgraph.exporters
from scipy import signal
import os
import pyqtgraph as pg
from scipy.fft import fftfreq, rfft, irfft
from numpy.fft import fft
import numpy as np
import sys
from PyQt5 import QtCore, QtWidgets, QtGui, uic
from PyQt5.QtCore import pyqtSlot, QSettings
from PyQt5.QtWidgets import QFileDialog, QMessageBox

import matplotlib
import matplotlib.ticker as ticker
from pdf import GeneratePDF
import img_rc  # for gui

matplotlib.use('Qt5Agg')


class AudioEqualizer(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi('AudioEqualizer.ui', self)

        self.settings = QSettings("Audio Equalizer", 'App')

        try:
            # Saving the last position and size of the application window
            self.resize(self.settings.value('window size'))
            self.move(self.settings.value('window position'))
        except:
            pass

        # Warning msg to prevent loading non-mono audio files
        self.warning_msg = QMessageBox()
        self.warning_msg.setWindowTitle("Warning")
        self.warning_msg.setText("Your file is not supported!")

        # Connecting Buttons
        self.action_open.triggered.connect(lambda: self.browse_file())
        self.action_new_win.triggered.connect(self.make_new_window)
        self.actionSave_as_PDF.triggered.connect(lambda: self.create_my_pdf())
        self.action_clear.triggered.connect(lambda: self.clear_all())
        self.actionClose.triggered.connect(lambda: self.close())
        self.show_ISpectroCh.clicked.connect(lambda: self.hide())

        self.actionPalette_1.triggered.connect(lambda: self.color_palette(0))
        self.actionPalette_2.triggered.connect(lambda: self.color_palette(1))
        self.actionPalette_3.triggered.connect(lambda: self.color_palette(2))
        self.actionPalette_4.triggered.connect(lambda: self.color_palette(3))
        self.actionPalette_5.triggered.connect(lambda: self.color_palette(4))

        self.InputCh.triggered.connect(
            lambda: self.select_channel(self.InputCh))
        self.ISpectroCh.triggered.connect(
            lambda: self.select_channel(self.ISpectroCh))
        self.OutputCh.triggered.connect(
            lambda: self.select_channel(self.OutputCh))
        self.OSpectroCh.triggered.connect(
            lambda: self.select_channel(self.OSpectroCh))

        self.right_button.clicked.connect(lambda: self.scroll_right())
        self.left_button.clicked.connect(lambda: self.scroll_left())
        self.up_button.clicked.connect(lambda: self.scroll_up())
        self.down_button.clicked.connect(lambda: self.scroll_down())

        self.zoom_in.clicked.connect(lambda: self.zoomin())
        self.zoom_out.clicked.connect(lambda: self.zoomout())

        self.Play_Button.clicked.connect(lambda: self.play())
        self.Stop_Button.clicked.connect(lambda: self.stop())

        self.Slider_11.valueChanged.connect(
            lambda: self.spectrogram_range())  # MinSlider
        self.Slider_12.valueChanged.connect(
            lambda: self.spectrogram_range())  # MaxSlider

        self.pens = [pg.mkPen('r'), pg.mkPen('b'), pg.mkPen('g')]

        # default intensity values
        self.min_slider_intensity = 0.5
        self.max_slider_intensity = 1.0

        # Color palettes LUT
        self.colors_list = [[(self.min_slider_intensity, (255, 0, 0, 255)),
                             (self.max_slider_intensity, (0, 0, 255, 255)),
                             (0.0, (0, 0, 0, 255))],

                            [(self.min_slider_intensity, (0, 182, 188, 255)),
                             (self.max_slider_intensity, (246, 111, 0, 255)),
                             (0.0, (75, 0, 113, 255))],

                            [(self.min_slider_intensity, (234, 214, 28, 255)),
                             (self.max_slider_intensity, (215, 199, 151, 255)),
                             (0.0, (0, 0, 0, 255))],

                            [(self.min_slider_intensity, (102, 204, 255, 255)),
                             (self.max_slider_intensity, (255, 102, 204, 255)),
                             (0.0, (0, 0, 0, 255))],

                            [(self.min_slider_intensity, (0, 149, 182, 255)),
                             (self.max_slider_intensity, (198, 195, 134, 255)),
                             (0.0, (0, 0, 0, 255))],

                            [(self.min_slider_intensity, (242, 226, 205, 255)),
                             (self.max_slider_intensity, (166, 158, 176, 255)),
                             (0.0, (0, 0, 0, 255))]]

        self.sliders_list = [self.Slider_1, self.Slider_2, self.Slider_3, self.Slider_4, self.Slider_5,
                             self.Slider_6, self.Slider_7, self.Slider_8, self.Slider_9, self.Slider_10]

        self.plotWidgets_list = [
            self.InputSignal, self.OutputSignal, self.InputSpectro, self.OutputSpectro]
        self.channels_list = [self.InputCh,
                              self.OutputCh, self.ISpectroCh, self.OSpectroCh]

        for i in range(10):
            self.sliders_list[i].valueChanged.connect(lambda: self.equalizer())
        self.mbands = []
        self.fbands = []
        self.gain = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # Related to QSettings

    def closeEvent(self, event):
        self.settings.setValue('window size', self.size())
        self.settings.setValue('window position', self.pos())

    # Creating new window by calling the main class

    def make_new_window(self):
        self.new_win = AudioEqualizer()
        self.new_win.show()

    # Open (.wav ) file, read it using Scipy Lib, and plot it in inputSignal Viewer

    def browse_file(self):
        self.selected_file = QtGui.QFileDialog.getOpenFileName(
            self, 'Select .wav file ', './', "Raw Data(*.wav)",  os.getenv('HOME'))

        path = str(self.selected_file[0])

        # get file extension
        self.file_ext = self.get_extention(path)
        # check the file extension is (.Wav)
        if self.file_ext == 'wav':
            # Read selected wav file
            self.samplerate, self.data = wavfile.read(path)

            self.length = self.data.shape[0]  # number of samples
            # The duration is equal to the number of frames divided by the framerate (frames per second)
            self.duration = (self.length / self.samplerate)

            # Return evenly spaced numbers over a specified interval
            self.time = np.linspace(0., self.duration, self.length)

            # self.freq --> The Discrete Fourier Transform sample frequencies.
            self.freq = fftfreq(self.length)

            if np.ndim(self.data) == 1:
                self.InputSignal.setYRange(10, -10)
                self.OutputSignal.setYRange(10, -10)

                self.InputSignal.setLimits(
                    xMin=0, xMax=500000, yMin=-200000, yMax=200000)

                self.OutputSignal.setLimits(
                    xMin=0, xMax=500000, yMin=-200000, yMax=200000)

                self.plot_spectrogram(
                    self.data, self.InputSpectro, self.colors_list[1])

                self.InputSignal.plot(self.time, self.data, pen=pg.mkPen('r'))
                self.OutputSignal.plot(self.time, self.data, pen=pg.mkPen('y'))
            # Our application does not support multi-channel wav files. only mono audio files (1channel)
            elif np.ndim(self.data) != 1:
                x = self.warning_msg.exec_()

            self.Slider_11.setMinimum(0)
            self.Slider_11.setMaximum(100)
            self.Slider_12.setMinimum(0)
            self.Slider_12.setMaximum(100)

        else:
            QMessageBox.warning(self.centralWidget,
                                'you must select .wav file')

    # plotting spectrogram into input and output spectrogram channels

    def plot_spectrogram(self, data_col, viewer, color, fs=44100):

        fs = self.samplerate
        # make sure the data given in array form
        data = np.array(data_col)

        # f : Array of sample frequencies; t : Array of segment times; Sxx : Spectrogram of x. The last axis of Sxx corresponds to the segment times.
        self.f, self.t, self.Sxx = signal.spectrogram(data, fs)

        # Interpret image data as row-major instead of col-major
        pyqtgraph.setConfigOptions(imageAxisOrder='row-major')
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
        hist.setLevels(np.min(self.Sxx), np.max(self.Sxx))
        # Sxx contains the amplitude for each pixel

        img.setImage(self.Sxx)
        # Scale the X and Y Axis to time and frequency (standard is pixels)
        img.scale(self.t[-1]/np.size(self.Sxx, axis=1),
                  self.f[-1]/np.size(self.Sxx, axis=0))
        # Limit panning/zooming to the spectrogram
        viewer.setLimits(xMin=0, xMax=self.t[-1], yMin=0, yMax=self.f[-1])

        # Range is limited to see the immediate changes easier
        viewer.setXRange(0, self.t[-1], padding=0)
        viewer.setYRange(0, self.f[-1], padding=0)

        # Add labels to the axis
        viewer.setLabel('bottom', "Time", units='s')
        # Include the units automatimy_pdfy scales the axis and adjusts the SI prefix (in this case kHz)
        viewer.setLabel('left', "Frequency", units='Hz')

        ################# Coloring Spectrogram ############
        hist.gradient.restoreState({'mode': 'rgb',
                                    'ticks': color})
        hist.gradient.showTicks(False)
        hist.shape
        hist.layout.setContentsMargins(0, 0, 0, 0)
        hist.vb.setMouseEnabled(x=False, y=False)

        ''' Another method to color the image by using *bipolar colormap* #just in case the 1st method failed

        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0,255,255,255], [255,255,0,255], [
                         0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        img.setLookupTable(lut)
        img.setLevels(np.min(Sxx), np.max(Sxx))

        '''

    def spectrogram_range(self):
        # changing slider values will change intensity in the palettes
        # intensity -> [0,1]
        self.min_slider_intensity = (float(self.Slider_11.value())/100)
        self.max_slider_intensity = (float(self.Slider_12.value())/100)

        # Updating old color LUT with new one holding new intensity values
        self.colors_list = [[(self.min_slider_intensity, (255, 0, 0, 255)),
                             (self.max_slider_intensity, (0, 0, 255, 255)),
                             (0.0, (0, 0, 0, 255))],

                            [(self.min_slider_intensity, (0, 182, 188, 255)),
                             (self.max_slider_intensity, (246, 111, 0, 255)),
                             (0.0, (75, 0, 113, 255))],

                            [(self.min_slider_intensity, (234, 214, 28, 255)),
                             (self.max_slider_intensity, (215, 199, 151, 255)),
                             (0.0, (0, 0, 0, 255))],

                            [(self.min_slider_intensity, (102, 204, 255, 255)),
                             (self.max_slider_intensity, (255, 102, 204, 255)),
                             (0.0, (0, 0, 0, 255))],

                            [(self.min_slider_intensity, (0, 149, 182, 255)),
                             (self.max_slider_intensity, (198, 195, 134, 255)),
                             (0.0, (0, 0, 0, 255))],

                            [(self.min_slider_intensity, (242, 226, 205, 255)),
                             (self.max_slider_intensity, (166, 158, 176, 255)),
                             (0.0, (0, 0, 0, 255))]]
        # Re-plotting the spectrogram with new color ranges
        self.plot_spectrogram(self.inversed_data,
                              self.OutputSpectro, self.colors_list[1])

    def color_palette(self, i):
        color = self.colors_list[i]

        if self.ISpectroCh.isChecked():
            self.plot_spectrogram(self.data, self.InputSpectro, color)

        if self.OSpectroCh.isChecked():
            self.plot_spectrogram(self.inversed_data,
                                  self.OutputSpectro, color)

    def select_channel(self, channel):

        if channel.isChecked():
            channel.setChecked(True)
        else:
            channel.setChecked(False)

    def hide(self):

        if (self.show_ISpectroCh.isChecked()):
            self.InputSpectro.hide()
        else:
            self.InputSpectro.show()

    def get_extention(self, s):
        for i in range(1, len(s)):
            if s[-i] == '.':
                return s[-(i - 1):]


#*******************************************Generating PDF**************************************#

    def create_my_pdf(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, 'Export PDF', None, 'PDF files (.pdf);;All Files()')

        # add ".pdf" to the name    "suffix"
        if file_name != '':

            if QtCore.QFileInfo(file_name).suffix() == "":
                file_name += '.pdf'

        # export all items in all viewers as images
        for i in range(len(self.plotWidgets_list)):
            exporter = pg.exporters.ImageExporter(
                self.plotWidgets_list[i].scene())
            widget_name = self.plotWidgets_list[i].objectName()
            exporter.export(f"{widget_name}.png")

        my_pdf = GeneratePDF(file_name)
        my_pdf.create_pdf()
        my_pdf.save_pdf()

#*******************************************Fourrier**************************************#
    def get_fft(self):
        # fft returns an array contains all +ve values then all -ve values
        # it has some real and some complex values
        self.fftArray = fft(self.data)
        # get the magnitude of both +ve & -ve
        self.fftArrayAbs = np.abs(self.fftArray)
        # get the phase
        self.fftPhase = np.angle(self.fftArray)
        # magnitude of +ve only
        self.fftMagnitude = self.fftArrayAbs[: self.length // 2]

        # return the magnitude and phase
        return self.fftArrayAbs, self.fftPhase


#*********************************************Equalizer***************************************#

    def equalizer(self):
        self.OutputSignal.clear()

        self.OutputSpectro.clear()

        self.fMagnitude = self.get_fft()[0]

        self.fPhase = self.get_fft()[1]

        self.mvaluePerBand = int(len(self.fMagnitude)/10)

        self.fvaluePerBand = int(len(self.freq)/10)

        self.newMagnitude = []

        self.outputSignal = []

        for i in range(10):
            self.mbands.append(
                self.fMagnitude[int(i * len(self.fMagnitude) / 10):int((i+1) * len(self.fMagnitude) / 10)])
            self.fbands.append(
                self.freq[int(i * len(self.freq) / 10):int((i+1) * len(self.freq) / 10)])

        for i in range(10):
            self.gain[i] = self.sliders_list[i].value()

        for index in range(10):
            # we changed it to np.array so we can multiply the value by value not multipling the list that will generate repetation of value not multplication

            Magnified_Magnitued = np.multiply(
                self.gain[index], (self.mbands[index]))
            self.newMagnitude.append(Magnified_Magnitued)

        for band in self.newMagnitude:
            for magnitude in band:
                self.outputSignal.append(magnitude)

        # self.fPhase == fftPhase
        finalSignal = np.multiply(
            np.exp(1j * self.fPhase), self.outputSignal)
        self.inversed_data = np.fft.irfft(finalSignal, len(self.fMagnitude))

        self.OutputSignal.plot(
            self.time, self.inversed_data, pen=pg.mkPen('y'))

        self.plot_spectrogram(
            self.inversed_data, self.OutputSpectro, self.colors_list[1])
        ######## To use Mag instead of phase: #######

        # Replace self.fPhase with self.outputSignal, and remove finalSignal

        # self.inversed_data = np.fft.irfft(self.outputSignal, len(self.fMagnitude))


#**********************************************toolbar********************************************#

    def play(self):
        if self.InputCh.isChecked():
            sd.play(self.data, self.samplerate)

        elif self.OutputCh.isChecked():
            sd.play(self.inversed_data)

    def stop(self):
        if self.InputCh.isChecked():
            sd.stop()

        elif self.OutputCh.isChecked():
            sd.stop()

    def clear_all(self):
        # set all sliders' value to 1
        for i in range(10):
            self.sliders_list[i].setProperty("value", 1)
        # clear all widgets
        for i in range(4):
            self.plotWidgets_list[i].clear()

    def zoomin(self):
        for i in range(4):
            if self.channels_list[i].isChecked():
                self.plotWidgets_list[i].plotItem.getViewBox().scaleBy(
                    (0.5, 0.5))

    def zoomout(self):
        for i in range(4):
            if self.channels_list[i].isChecked():
                self.plotWidgets_list[i].plotItem.getViewBox().scaleBy(
                    (1.5, 1.5))

# scrolling only works with input-signal and output-signal widgets
    def scroll_right(self):
        for i in range(2):
            self.range = self.plotWidgets_list[i].getViewBox().viewRange()
            if self.range[0][1] < max(self.time):
                self.plotWidgets_list[i].getViewBox().translateBy(x=+0.2, y=0)

    def scroll_left(self):
        for i in range(2):
            self.range = self.plotWidgets_list[i].getViewBox().viewRange()
            if self.range[0][0] > min(self.time):
                self.plotWidgets_list[i].getViewBox().translateBy(x=-0.2, y=0)

    def scroll_up(self):
        for i in range(2):
            self.range = self.plotWidgets_list[i].getViewBox().viewRange()
            if self.range[1][1] < max(self.data):
                self.plotWidgets_list[i].getViewBox().translateBy(x=0, y=+0.2)

    def scroll_down(self):
        for i in range(2):
            self.range = self.plotWidgets_list[i].getViewBox().viewRange()
            if self.range[1][0] > min(self.data):
                self.plotWidgets_list[i].getViewBox().translateBy(x=0, y=-0.2)


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
