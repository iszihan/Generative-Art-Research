import os,sys
import argparse
import ast
import csv
from PyQt4 import QtGui,QtCore

scaled_dim = 40 #the size of the image on the visulization window
zoom_dim = 300 #the size when the mouse is over
parser = argparse.ArgumentParser()
parser.add_argument('directories', action='store', help='The directories to load images from',default=None)
# parser.add_argument('-m', '--parameters', action='store', dest='parameters', default=None)
parser.add_argument('-p', '--number of parameters', action='store', type=int, dest='n_parameters', default=1)
parser.add_argument('-n', '--selection of parameter', action='store', type=int, dest='parameter', default=0)
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
    value = []

    max_value = [0.0,0.0]
    min_value = [10.0,10.0]
    range_value = [0.0,0.0]



    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, self.width, self.height)
        self.read_file(args.directories, args.n_parameters)
        self.add_images(args.n_parameters,args.parameter)
        # shot_button = QtGui.QPushButton("Screenshot",self)
        # shot_button.move(self.width/2,self.height/2)
        # shot_button.clicked.connect(self.shoot)
        self.show()



    def read_file(self,dir,n_param):
        f = open(os.path.join(dir,'PredictResults.csv'),'r')
        has_header = csv.Sniffer().has_header(f.readline())
        readFile = csv.reader(f)
        if has_header:
            next(readFile)
        for row in readFile: #read the csv file
            self.FilePath.append(row[0])
            for i in range(n_param):
                self.value.append([])
                self.value[i].append(row[i+1])
                if(float(row[i+1])>self.max_value[i]):
                    self.max_value[i] = float(row[i+1])
                if(float(row[i+1])<self.min_value[i]):
                    self.min_value[i] = float(row[i+1])

        print("max:",self.max_value)
        print("min:",self.min_value)
        for i in range(len(self.max_value)):
            self.range_value[i] = self.max_value[i]-self.min_value[i]
        print("range:",self.range_value)

    def add_images(self,n_param,param):
        for i in range(len(self.FilePath)-1):
            print(self.FilePath[i+1])
            if(n_param == 2):
                if(param==0):
                    temp_x = float(self.value[0][i+1])*(self.width/self.max_value[0])-100
                    temp_y = float(self.value[1][i+1])*(self.height/self.max_value[1])
                    pic = ImageButton(self)
                    pic.setIcon(QtGui.QIcon(os.path.join('Images',self.FilePath[i+1])))
                    pic.setIconSize(QtCore.QSize(scaled_dim,scaled_dim))
                    pic.move(temp_x,self.height-temp_y)
                    pic.setStyleSheet('border: none')
                elif(param==1):
                    temp_x = float(self.value[0][i+1])*(self.width/self.max_value[0])
                    pic = ImageButton(self)
                    pic.setIcon(QtGui.QIcon(os.path.join('Images',self.FilePath[i+1])))
                    pic.setIconSize(QtCore.QSize(scaled_dim,scaled_dim))
                    pic.move(temp_x,self.height/2)
                    pic.setStyleSheet('border: none')
                elif(param==2):
                    temp_x = float(self.value[1][i+1])*(self.width/self.max_value[1])
                    pic = ImageButton(self)
                    pic.setIcon(QtGui.QIcon(os.path.join('Images',self.FilePath[i+1])))
                    pic.setIconSize(QtCore.QSize(scaled_dim,scaled_dim))
                    pic.move(temp_x,self.height/2)
                    pic.setStyleSheet('border: none')
            elif(n_param == 1):
                    temp_x = float(self.value[0][i+1])*(self.width/self.max_value[0])
                    pic = ImageButton(self)
                    pic.setIcon(QtGui.QIcon(os.path.join('Images',self.FilePath[i+1])))
                    pic.setIconSize(QtCore.QSize(scaled_dim,scaled_dim))
                    pic.move(temp_x,self.height/2)
                    pic.setStyleSheet('border: none')


    # def shoot(self):
    #     p = QtGui.QPixmap.grabWidget(self,0,0,self.width,self.height)
    #     filename =
    #     p.save(filename, 'JPG')
    #     print("shot taken")



app = QtGui.QApplication(sys.argv)
window = Visualization()
sys.exit(app.exec_())
