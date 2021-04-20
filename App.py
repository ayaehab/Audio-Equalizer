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
        self.ui = uic.loadUi('newGUI.ui', self)

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
        self.show_ISpectroCh.clicked.connect(lambda : self.hide())


        self.actionPalette_1.triggered.connect(lambda: self.color_palette(0))
        self.actionPalette_2.triggered.connect(lambda: self.color_palette(1))
        self.actionPalette_3.triggered.connect(lambda: self.color_palette(2))
        self.actionPalette_4.triggered.connect(lambda: self.color_palette(3))
        self.actionPalette_5.triggered.connect(lambda: self.color_palette(4))

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

        self.colors_list = [[(0.5, (255, 0, 0, 255)),
                             (1.0, (0, 0,255, 255)),
                             (0.0, (0, 0, 0, 255))], [(0.5, (0, 182, 188, 255)),
                                                      (1.0, (246, 111, 0, 255)),
                                                      (0.0, (75, 0, 113, 255))],  [(0.5, (234, 214, 28, 255)),
                                                                                   (1.0, (215, 199,151, 255)),
                                                                                   (0.0, (0, 0, 0, 255))], [(0.5, (102, 204, 255, 255)),
                                                                                                            (1.0, (255, 102,204, 255)),
                                                                                                            (0.0, (0, 0, 0, 255))], [(0.5, (0, 149, 182, 255)),
                                                                                                                                     (1.0, (198, 195, 134, 255)),
                                                                                                                                     (0.0, (0, 0, 0, 255))], [(0.5, (242, 226, 205, 255)),
                                                                                                                                                              (1.0, (166, 158, 176, 255)),
                                                                                                                                                              (0.0, (0, 0, 0, 255))]]

        self.default_color = self.colors_list[1]

        self.sliderList = [self.Slider_1, self.Slider_2, self.Slider_3, self.Slider_4, self.Slider_5,
                           self.Slider_6, self.Slider_7, self.Slider_8, self.Slider_9, self.Slider_10]
        
        self.Slider_11.valueChanged.connect(
            lambda: self.spectrogram_range_limiter(self.inverse))  # MinSlider
        self.Slider_12.valueChanged.connect(
            lambda: self.spectrogram_range_limiter(self.inverse))  # MaxSlider

        for i in range(10):
            self.sliderList[i].valueChanged.connect(lambda: self.equalizer())

        self.mbands = []
        self.fbands = []
        self.gain = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    #Related to QSettings 
    def closeEvent(self, event):
        self.settings.setValue('window size', self.size())
        self.settings.setValue('window position', self.pos())

    #Creating new window by calling the main class 
    def make_new_window(self):
        self.new_win = AudioEqualizer()
        self.new_win.show()

    #plotting spectrogram into input and output spectrogram channels
    def plot_spectrogram(self, data_col, viewer, color, fs=44100):
        
        fs = self.samplerate
        # make sure the data given in array form
        data = np.array(data_col)

        # f : Array of sample frequencies; t : Array of segment times; Sxx : Spectrogram of x. The last axis of Sxx corresponds to the segment times.
        self.f, self.t, self.Sxx = signal.spectrogram(data, fs)
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
        # Most image data is stored in row-major order (row, column) and will need to be transposed before calling setImage():
        img.setImage(self.Sxx.T)
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

    def color_palette(self, i):
        color = self.colors_list[i]
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
                
    def hide(self ) :

        if (self.show_ISpectroCh.isChecked()) :
            self.InputSpectro.hide()
        else :
            self.InputSpectro.show()

# Open (.wav ) file, read it using Scipy Lib, and plot it in inputSignal Viewer
    def browse_file(self):
        self.selected_file = QtGui.QFileDialog.getOpenFileName(
            self, 'Select .wav file ', './', "Raw Data(*.wav)",  os.getenv('HOME'))

        path = str(self.selected_file[0])
        
        #get file extension
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

            #self.freq --> The Discrete Fourier Transform sample frequencies.
            self.freq = fftfreq(self.length)

            if np.ndim(self.data) == 1:
                self.InputSignal.setYRange(10, -10)
                self.OutputSignal.setYRange(10, -10)
                
                self.InputSignal.setLimits(
                    xMin=0, xMax=500000, yMin=-200000, yMax=200000)

                self.OutputSignal.setLimits(
                    xMin=0, xMax=500000, yMin=-200000, yMax=200000)

                self.plot_spectrogram(
                    self.data, self.InputSpectro, self.default_color)

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

    ##### Generating the PDF ####

    # this funcition for the button, when pressed 4 images exported, pdf generated, then the pictures get deleted

    def create_my_pdf(self):

        file_name, _ = QFileDialog.getSaveFileName(
            self, 'Export PDF', None, 'PDF files (.pdf);;All Files()')
        if file_name != '':

            # add ".pdf" to the name
            if QtCore.QFileInfo(file_name).suffix() == "":
                file_name += '.pdf'

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

            my_pdf = GeneratePDF(file_name)
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
        # get the magnitude of both +ve & -ve
        self.fftArrayAbs = np.abs(self.fftArray)
        # get the phase
        self.fftPhase = np.angle(self.fftArray)
        # magnitude of +ve only
        self.fftMagnitude = self.fftArrayAbs[: self.length // 2]
        
        # return the magnitude and phase
        return self.fftArrayAbs, self.fftPhase


#*****************************************END OF Fourrier*************************************#


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
            self.gain[i] = self.sliderList[i].value()

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
        self.inverse = np.fft.irfft(finalSignal, len(self.fMagnitude))


        self.OutputSignal.plot(self.time, self.inverse, pen=pg.mkPen('y'))
        self.plot_spectrogram(
            self.inverse, self.OutputSpectro, self.default_color)
        ######## To use Mag instead of phase: #######
        
        # Replace self.fPhase with self.outputSignal, and remove finalSignal
        
        # self.inverse = np.fft.irfft(self.outputSignal, len(self.fMagnitude))

#########################
#*******************************************End of Equalizer**************************************#

    def spectrogram_range_limiter(self, data_col):
        self.OutputSpectro.clear()
        
        if self.Slider_12.value() > self.Slider_11.value():
            fs = self.samplerate 
            spectro_min = ( float(self.Slider_11.value())/100 )
            spectro_max = ( float(self.Slider_12.value())/100 )
            
            #ReCompute the 1-D discrete Fourier Transform to limit the signal in the frequency domain
            fft_array = rfft(data_col)
            begin_index = int((len(fft_array)) * spectro_min) 
            end_index = int((len(fft_array)) * spectro_max) 
            fft_array[0:begin_index] = 0
            fft_array[end_index:len(fft_array)] = 0
            #Transform the data back into the time domain to plot it.
            self.output_array = irfft(fft_array)
            self.plot_spectrogram(self.output_array, self.OutputSpectro, self.default_color, fs=fs)

        else:
            #Show nothing if min_slider.value > max_slider.value
            self.OutputSpectro.clear()

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
        self.default()
        self.InputSignal.clear()
        self.InputSpectro.clear()
        self.OutputSignal.clear()
        self.OutputSpectro.clear()

    def default(self):
        for i in range(10):
            self.sliderList[i].setProperty("value", 1)

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
