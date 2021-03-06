"""
This PyQt4 visualization applet is used to spatialize the images according to their predictions generated by the Convolutional Neural Network Model.
Steps to use this applet:
1. Install PyQt4 by 'conda install pyqt=4'
2. Run Parameters.py's Predict function (See ReadMe.md) for images that you want, and save the result folder in the same directory as Vis.py.
3. Run this applet like below:
    To visualize results in PredictionCSVFolder that has only 1 parameter:
        python3 Vis.py PredictionCSVFolder -p 1
    To visualize results in PredictionCSVFolder that has 2 parameters and spatialize the images according to both parameters:
        python3 Vis.py PredictionCSVFolder -p 2
    To visualize results in PredictionCSVFolder that has 2 parameters and spatialize the images according to the first parameters:
        python3 Vis.py PredictionCSVFolder -p 2 -n 1
"""


import os,sys
import argparse
import ast
import csv
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt4 import QtGui,QtCore

scaled_dim = 150 #the size of the image on the visulization window
zoom_dim = 400 #the size when the mouse is over

parser = argparse.ArgumentParser()
parser.add_argument('directories', action='store', help='The directories to load images from',default=None)
parser.add_argument('-p', '--number of parameters', action='store', type=int, dest='n_parameters', default=1)
parser.add_argument('-n', '--selection of parameter', action='store', type=int, dest='parameter', default=0)
args = parser.parse_args()

print('Directories to load from:', os.path.abspath(os.path.join(args.directories, 'PredictResults.csv')))

class ImageButton(QtGui.QPushButton):


    def __init__(self,parent=None):
        super().__init__(parent)
        self.clicked.connect(self.enlarge)

    def enlarge(self):
        self.setIconSize(QtCore.QSize(zoom_dim,zoom_dim))
        self.resize(zoom_dim,zoom_dim)
        self.raise_()
        self.clicked.connect(self.zoomout)

    def zoomout(self):
        self.setIconSize(QtCore.QSize(scaled_dim,scaled_dim))
        self.resize(scaled_dim,scaled_dim)
        self.lower()
        self.clicked.connect(self.enlarge)


class Visualization(QtGui.QWidget):

    width = 0 #window size
    height = 0

    data = []
    values = []
    images = []

    def __init__(self,parent=None):
        super().__init__(parent)

        l=QtGui.QVBoxLayout(self)
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(0)

        content = QtGui.QWidget(self)
        if args.n_parameters==2 and args.parameter==0:
            self.width=2500
            self.height=1000
        else:
            self.width=2500
            self.height=450
        self.setGeometry(0, 0, self.width, self.height)

        self.read_file(args.directories, args.n_parameters, args.parameter)
        self.add_images(args.n_parameters,args.parameter,content)
        content.setGeometry(0, 0, self.width+400, self.height+400)

        s = QtGui.QScrollArea()
        l.addWidget(s)

        s.setWidget(content)



    def read_file(self,dir,n_param,param):
        f = open(os.path.join(dir,'PredictResults.csv'),'r')
        readFile = csv.reader(f)
        header = next(readFile)
        print("Header:",header)
        for row in readFile: #read the csv file from the second row
            single_row = []
            single_row.append(row[0])
            for i in range(n_param):
                single_row.append(float(row[i+1]))
            self.data.append(single_row)

        self.data = np.asarray(self.data)
        #single parameter to visualize
        if n_param == 2 and param != 0:
            print("Which parameter:",header[param][0])
            title = "Loaded from '"+ args.directories+"' for parameter "+header[param][0]
            self.setWindowTitle(title)
            self.data = self.data[self.data[:,param].argsort()]

            self.values = self.data[:,param].astype(np.float)
            self.images = self.data[:,0]
            self.images = self.images[self.values.argsort()]
            self.values = self.values[self.values.argsort()]

        elif n_param==1:
            print("Which parameter:",header[1][0])
            title = "Loaded from '"+ args.directories+"' for parameter "+header[1][0]
            self.setWindowTitle(title)

            self.values = self.data[:,1].astype(np.float)
            self.images = self.data[:,0]
            self.images = self.images[self.values.argsort()]
            self.values = self.values[self.values.argsort()]

        #two parameters to visualize
        elif n_param==2 and param==0:
            print("Which parameter:",header[1][0],header[2][0])
            title = "Loaded from '"+ args.directories+"' for parameter "+header[1][0]+"(x-axis) and "+header[2][0]+"(y-axis)"
            self.setWindowTitle(title)
            self.values = self.data[:,1:3].astype(np.float)
            self.images = self.data[:,0]
            self.images = self.images[self.values[:,0].argsort()]
            self.values = self.values[self.values[:,0].argsort()]




    def add_images(self,n_param,param,content):
        prev_x = 0
        curr_x = 0

        for i in range(len(self.data)):
            if(n_param == 2):
                #two parameter visualization
                if(param==0):
                    if i==0:
                        curr_x = self.values[i,0]*(2000/11)
                    else:
                        curr_x = prev_x+scaled_dim+(self.values[i,0]-self.values[i-1,0])*(2000/11) #Avoid overlapping
                    curr_y = float(self.values[i,1])*(self.height/self.values[:,1].max())

                    pic = ImageButton(content)
                    labelx="x="+str(self.values[i,0])
                    labely="y="+str(self.values[i,1])
                    pred_x = QtGui.QLabel(labelx,content) #To show the numerical prediction
                    pred_y = QtGui.QLabel(labely,content)
                    img = Image.open(os.path.join('Images',self.images[i]))
                    img = ImageQt(img)
                    img = QtGui.QPixmap.fromImage(img)
                    pic.setIcon(QtGui.QIcon(img))
                    pic.setIconSize(QtCore.QSize(scaled_dim,scaled_dim))

                    pic.move(curr_x,self.height-curr_y)
                    pred_x.move(curr_x,self.height-curr_y-24)
                    pred_y.move(curr_x,self.height-curr_y-12)
                    pic.setStyleSheet('border: none')

                    prev_x = curr_x #remember the preivous image's coordiante

                #single parameter visualization
                elif(param==1 or param==2): #first parameter
                    if i==0:
                        curr_x = self.values[i]*(2000/11)
                    else:
                        curr_x = prev_x+scaled_dim+(self.values[i]-self.values[i-1])*(2000/11) #Avoid overlapping
                    pic = ImageButton(content)
                    pred = QtGui.QLabel(str(self.values[i]),content) #To show the numerical prediction
                    img = Image.open(os.path.join('Images',self.images[i]))
                    img = ImageQt(img)
                    img = QtGui.QPixmap.fromImage(img)
                    pic.setIcon(QtGui.QIcon(img))
                    pic.setIconSize(QtCore.QSize(scaled_dim,scaled_dim))
                    pic.move(curr_x,50)
                    pred.move(curr_x,38)
                    pic.setStyleSheet('border: none')
                    prev_x = curr_x


            #single parameter visualization
            elif(n_param == 1):
                    if i==0:
                        curr_x = 10 #self.values[i]*(2000/11)
                    else:
                        curr_x = prev_x+scaled_dim+(self.values[i]-self.values[i-1])*(2000/11) #Avoid overlapping
                    pic = ImageButton(content)
                    pred = QtGui.QLabel(str(self.values[i]),content) #To show the numerical prediction
                    img = Image.open(os.path.join('Images',self.images[i]))
                    img = ImageQt(img)
                    img = QtGui.QPixmap.fromImage(img)
                    pic.setIcon(QtGui.QIcon(img))
                    pic.setIconSize(QtCore.QSize(scaled_dim,scaled_dim))
                    pic.move(curr_x,50)
                    pic.setStyleSheet('border: none')
                    pred.move(curr_x,38)
                    prev_x = curr_x

            self.width = prev_x #set the scrollable widget width to be the x-coordinate of the last-added image, i.e. the image with the max parameter value



app = QtGui.QApplication(sys.argv)
window = Visualization()
window.show()
window.raise_()
sys.exit(app.exec_())
