import os,sys
import argparse
import ast
import csv
from PyQt4 import QtGui,QtCore

scaled_dim = 40 #the size of the image on the visulization window
zoom_dim = 300 #the size when the mouse is over
parser = argparse.ArgumentParser()
parser.add_argument('directories', action='store', help='The directories to load images from',default=None)
args = parser.parse_args()

print('Directories to load from:', os.path.abspath(os.path.join(args.directories, 'PredictResults.csv')))

class ImageButton(QtGui.QPushButton):


    def __init__(self,parent=None):
        super().__init__(parent)
        # self.setMouseTracking(True)
        self.clicked.connect(self.enlarge)

    # def enterEvent(self,event):
    #     self.setIconSize(QtCore.QSize(zoom_dim,zoom_dim))
    #     self.raise_()
    #
    # def leaveEvent(self,event):
    #     self.setIconSize(QtCore.QSize(scaled_dim,scaled_dim))

    def enlarge(self):
        self.setIconSize(QtCore.QSize(zoom_dim,zoom_dim))
        self.resize(zoom_dim,zoom_dim)
        self.raise_()
        self.clicked.connect(self.zoomout)

    def zoomout(self):
        self.setIconSize(QtCore.QSize(scaled_dim,scaled_dim))
        self.resize(scaled_dim,scaled_dim)
        # pic.setGeometry(temp_x,temp_y,scaled_dim,scaled_dim)
        self.lower()
        self.clicked.connect(self.enlarge)



class Visualization(QtGui.QWidget):

    width = 1000 #window size
    height = 700

    FilePath = [] #list to store values extracted from the csv file
    r_value = []
    m_value = []

    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, self.width, self.height)
        self.read_file(args.directories)
        self.add_images()
        self.show()



    def read_file(self,dir):
        f = open(os.path.join(dir,'PredictResults.csv'),'r')
        readFile = csv.reader(f)
        for row in readFile: #read the csv file
            self.FilePath.append(row[0])
            self.r_value.append(row[1])
            self.m_value.append(row[2])

    def add_images(self):
        for i in range(len(self.FilePath)-1):
            print(self.FilePath[i+1])
            temp_x = float(self.r_value[i+1])*(self.width/100)-100
            temp_y = float(self.m_value[i+1])*(self.height/100)
            pic = ImageButton(self)

            pic.setIcon(QtGui.QIcon(os.path.join('Images',self.FilePath[i+1])))
            pic.setIconSize(QtCore.QSize(scaled_dim,scaled_dim))
            pic.move(temp_x,temp_y)
            pic.setStyleSheet('border: none')



app = QtGui.QApplication(sys.argv)
window = Visualization()
sys.exit(app.exec_())
