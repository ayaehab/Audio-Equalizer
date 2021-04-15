from reportlab.pdfgen import canvas
import os
class GeneratePDF():

    def __init__(self):
        super().__init__()
    # Content for PDF
        self.fileName = "Audio Equalizer.pdf"
        self.documentTitle = 'Equalizer'
        self.title = 'Equalizer'

        # Signals imgs used in generating PDF
        self.input_signal = 'input_signal.png'
        self.input_spectro = 'input_spectro.png'
        self.output_signal = 'output_signal.png'
        self.output_spectro = 'output_spectro.png'

        #  Create document with content given

        self.pdf = canvas.Canvas(self.fileName)
        self.pdf.setTitle(self.documentTitle)

        #  Adjusting title

        self.pdf.setFont('Courier-Bold', 36)
        self.pdf.drawCentredString(315, 795, 'Audio Equalizer')

        #  Adjusting sub-title

        self.pdf.setFont('Courier-Bold', 12)
        # self.pdf.drawString(315, 795, 'Signal')
        

        #  Draw all lines for the table
        self.pdf.line(10, 780, 570, 780)
        self.pdf.line(10, 580, 570, 580)
        self.pdf.line(10, 380, 570, 380)
        self.pdf.line(10, 180, 570, 180)

        self.pdf.line(110, 20, 110, 800)
        

    # ###################################
    ##############################
    def drawMyRuler(self):
        self.pdf.drawString(100, 810, 'x100')
        self.pdf.drawString(200, 810, 'x200')
        self.pdf.drawString(300, 810, 'x300')
        self.pdf.drawString(400, 810, 'x400')
        self.pdf.drawString(500, 810, 'x500')

        self.pdf.drawString(10, 100, 'y100')
        self.pdf.drawString(10, 200, 'y200')
        self.pdf.drawString(10, 300, 'y300')
        self.pdf.drawString(10, 400, 'y400')
        self.pdf.drawString(10, 500, 'y500')
        self.pdf.drawString(10, 600, 'y600')
        self.pdf.drawString(10, 700, 'y700')
        self.pdf.drawString(10, 800, 'y800')
    ######################

# Plotting the signals names
    def sigName(self, signal1, signal2, signal3 ,signal4):
        self.pdf.drawString(2, 700, signal1)
        self.pdf.drawString(2, 500, signal2)
        self.pdf.drawString(2, 300, signal3)
        self.pdf.drawString(2, 100, signal4)

# Sending all signals images to their positions in the table
    def sigImage(self, img1, img2):
        self.pdf.drawInlineImage(img1, 120, 595, width=410,
                                 height=170, preserveAspectRatio=False, showBoundary=True)
        self.pdf.drawInlineImage(img2, 120, 195, width=410,
                                 height=170, preserveAspectRatio=False, showBoundary=True)



# Sending all signals spectroimages to their positions in the table
    def spectroImage(self, img1, img2):
        self.pdf.drawInlineImage(img1, 120, 395, width=410,
                                 height=170,  preserveAspectRatio=False, showBoundary=True)
        self.pdf.drawInlineImage(img2, 120, 5, width=410,
                                 height=170, preserveAspectRatio=False, showBoundary=True)

    # Generating the PDF

    def create_pdf(self):
        self.sigName('Input Signal', 'I-Spectrogram', 'Output Signal', 'O-Spectrogram') #reduce font size 
        self.sigImage(self.input_signal, self.output_signal)
        self.spectroImage(self.input_spectro, self.output_spectro)
        # self.drawMyRuler()
        self.save_pdf()

    def save_pdf(self):
        self.pdf.save()
        
        # delete created images after generating PDF file 
        if os.path.exists("Audio Equalizer.pdf"):
            os.remove("input_signal.png")
            os.remove("output_signal.png")
            os.remove("input_spectro.png")
            os.remove("output_spectro.png")


# To run the code:
# test = GeneratePDF()
# test.create_pdf()
