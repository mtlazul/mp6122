"""
GUI based on Eli Bendersky's (eliben@gmail.com) pyQt4 + matplotlib demo

Miguel Taylor (mtlazul@gmail.com)
License: this code is in the public domain
"""
import wormLines as wl
import numpy as np

import sys, os, random, re, cv2, matplotlib, json
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from datetime import datetime

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
     

class AppForm(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Wormlines')
        self.imagePaths = []
        self.dataPaths = []
        self.currentNematodeIndex = 0

        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()

        
        #self.imagePaths = ['/home/taylor/Documents/tfg/BBDD_2013/BBDD_sinprocesar/BBDD/image/file1/segmentados/f1-12.png']
        #self.dataPaths = ['/home/taylor/Documents/tfg/BBDD_2013/BBDD_sinprocesar/BBDD/landmarks/file1/f1-12.dat']
        #self.currentNematodeIndex = 1
        #self.show_currentNematodeIndex()
        #self.save_json()
        
        self.canvas.draw()

        
    def create_menu(self):
        # File submenu
        self.file_menu = self.menuBar().addMenu("&File")
        load_data_action = self.create_action("&Load data", shortcut="Ctrl+L", slot=self.load_data, tip="Load landmarks data")
        open_image_action = self.create_action("&Open image", shortcut="Ctrl+O", slot=self.open_image, tip="Open nematodes images")
        save_json_action = self.create_action("&Save as JSON", shortcut="Ctrl+S", slot=self.save_json, tip="Save the edited splines as a JSON")
        batch_action = self.create_action("&Batch process", slot=self.batch_process, shortcut="Ctrl+P", tip="Process all images with current parameters")
        quit_action = self.create_action("&Quit", slot=self.close, shortcut="Ctrl+Q", tip="Close the application")
        self.add_actions(self.file_menu, (load_data_action, open_image_action, save_json_action, batch_action, quit_action))

        # Navigation submenu
        self.navigation_menu = self.menuBar().addMenu("&Navigation")
        next_nematode_action = self.create_action("&Next nematode", shortcut="Ctrl+Right", slot=self.next_nematode, tip="Plot the next pre-loaded nematode")
        previous_nematode_action = self.create_action("&Previous nematode", shortcut="Ctrl+Left", slot=self.previous_nematode, tip="Plot the previous pre-loaded nematode")
        shift_landmark_up_action = self.create_action("&Increment landmark index", shortcut="Ctrl+Up", slot=self.inc_landmark, tip="Increment landmark position in the list")
        shift_landmark_down_action = self.create_action("&Drecrement landmark index", shortcut="Ctrl+Down", slot=self.dec_landmark, tip="Decrement landmark position in the list")
        self.add_actions(self.navigation_menu, (next_nematode_action, previous_nematode_action, shift_landmark_up_action, shift_landmark_down_action))
        
        # Help submenu
        self.help_menu = self.menuBar().addMenu("&Help")
        help_action = self.create_action("&Wormlines help", slot=self.on_help, tip="Wormlines help")
        about_action = self.create_action("&About", shortcut='F1', slot=self.on_about, tip='About Wormlines')
        self.add_actions(self.help_menu, (help_action,about_action))

    def open_image(self):
        self.imagePaths = []
        for path in QFileDialog.getOpenFileNames(self,"Select one or more images to open","/home/taylor/Documents/tfg/BBDD_2013/BBDD_sinprocesar/BBDD","Images (*.png)"):
            self.imagePaths.append(path)
        self.statusBar().showMessage('Loaded %d images' % len(self.imagePaths))
        self.axes.clear()
        if self.imagePaths:
            self.currentNematodeIndex = 1
            self.show_currentNematodeIndex()
        else:
            self.currentNematodeIndex = 0
            self.slider.setVisible(False)
            self.textbox.setVisible(False)
            self.statusBar().showMessage("Showing image "+str(self.currentNematodeIndex)+"/"+str(len(self.imagePaths)))

    def load_data(self):
        self.dataPaths = []
        for path in QFileDialog.getOpenFileNames(self,"Select one or more files to open","/home/taylor/Documents/tfg/BBDD_2013/BBDD_sinprocesar/BBDD","Data (*.dat)"):
            self.dataPaths.append(path)
        self.statusBar().showMessage('Loaded %d data files' % len(self.dataPaths))
        self.axes.clear()
        if self.imagePaths:
            self.show_currentNematodeIndex()
        else:
            self.slider.setVisible(False)
            self.textbox.setVisible(False)

    def show_currentNematodeIndex(self):
        filenameMatch = re.findall("([^/]+)\.png$", self.imagePaths[self.currentNematodeIndex-1])
        regex = re.compile(".*"+str(filenameMatch[0])+"\.dat")
        dataPath = filter(regex.match, self.dataPaths)
        if dataPath:
            pngRoute = str(self.imagePaths[self.currentNematodeIndex-1])
            datRoute = str(dataPath[0])
            self.currentNematode = wl.nematode(pngRoute, datRoute,self.axes, int(self.textbox.text()))
            self.selected, = self.axes.plot(0, 0, 'co', ms = 0)
            self.canvas.draw()
            self.slider.setVisible(True)
            self.textbox.setVisible(True)
            self.statusBar().showMessage("Showing image "+str(self.currentNematodeIndex)+"/"+str(len(self.imagePaths)))
        else:
            image = cv2.imread(str(self.imagePaths[self.currentNematodeIndex-1]), cv2.IMREAD_GRAYSCALE)
            rows,cols = image.shape
            self.axes.imshow(image,cmap = 'gray', interpolation='nearest',  picker=False)
            self.axes.axis([0, cols, rows, 0])
            self.axes.axis('off')
            self.canvas.draw()
            self.slider.setVisible(False)
            self.textbox.setVisible(False)
            self.statusBar().showMessage('Data file for %s is missing' % (str(filenameMatch[0])+".png"))

    def next_nematode(self):
        self.axes.clear()
        if self.currentNematodeIndex == 0:
            return
        self.currentNematodeIndex += 1
        if self.currentNematodeIndex > len(self.imagePaths):
            self.currentNematodeIndex = 1
        self.show_currentNematodeIndex()
            
    def previous_nematode(self):
        self.axes.clear()
        if self.currentNematodeIndex == 0:
            return
        self.currentNematodeIndex -= 1
        if self.currentNematodeIndex < 1:
            self.currentNematodeIndex = len(self.imagePaths)
        self.show_currentNematodeIndex()

    def inc_landmark(self):
        self.currentNematode.shiftLandmarkPosition (self.selected.get_data(), 1)
        self.currentNematode.updatePlot()
        self.canvas.draw()

    def dec_landmark(self):
        self.currentNematode.shiftLandmarkPosition (self.selected.get_data(), -1)
        self.currentNematode.updatePlot()
        self.canvas.draw()
            
    def save_json(self):
        if self.imagePaths:
            filenameMatch = re.findall("([^/]+)\.png$", self.imagePaths[self.currentNematodeIndex-1])
            path = os.getcwd()+"/json/"+str(filenameMatch[0])+".json"
            jsonData = self.currentNematode.serialize()
            newFile = open(path,"w")
            newFile.write(jsonData)
            newFile.close()
            self.statusBar().showMessage('Saved to %s' % path)
        else:
            self.statusBar().showMessage('No loaded data')

    def batch_process(self):
        if self.imagePaths:
            logPath = os.getcwd()+"/log.json"
            try:
                logFile = open(logPath,"r")
                logData = json.loads(logFile.read())
                errorImagePaths = logData["errorImagePaths"]
                errorDataPaths = logData["errorDataPaths"]
                logFile.close()
            except:
                errorImagePaths = []
                errorDataPaths = []
            for imagePath in self.imagePaths:
                filenameMatch = re.findall("([^/]+)\.png$", imagePath)
                regex = re.compile(".*"+str(filenameMatch[0])+"\.dat")
                dataPath = filter(regex.match, self.dataPaths)
                if dataPath:
                    datRoute = str(dataPath[0])
                    try:
                        self.currentNematode = wl.nematode(str(imagePath), datRoute, self.axes, int(self.textbox.text()))
                        if wl.hasIntersections(self.currentNematode.points):
                            errorImagePaths += [str(imagePath)]
                            errorDataPaths += [datRoute]
                            self.statusBar().showMessage("Err fixing file: %s" %str(filenameMatch[0]))
                        else:
                            self.statusBar().showMessage("Processing file: %s" %str(filenameMatch[0]))
                            self.currentNematode.adjustLandmarks(self.landmarksValue)
                            self.currentNematode.adjustSpline(self.splineValue)
                            path = os.getcwd()+"/json/"+str(filenameMatch[0])+".json"
                            jsonData = self.currentNematode.serialize()
                            newFile = open(path,"w")
                            newFile.write(jsonData)
                            newFile.close()
                    except:
                        errorImagePaths += [str(imagePath)]
                        errorDataPaths += [datRoute]
                        self.statusBar().showMessage("Error with file: %s" %str(filenameMatch[0]))
                else:
                    self.statusBar().showMessage("Missing dat file %s" %str(filenameMatch[0]))
            logFile = open(logPath,"w")
            logData = {"errorImagePaths": errorImagePaths,
                    "errorDataPaths": errorDataPaths}
            logJsonData = json.dumps(logData)
            logFile.write(logJsonData)
            logFile.close()
            self.statusBar().showMessage("Finished processing")
        else:
            self.statusBar().showMessage('No loaded data')
        

    def on_help(self):
        msg = """ TODO """
        QMessageBox.about(self, "Help", msg.strip())
    
    def on_about(self):
        msg = """ TODO """
        QMessageBox.about(self, "About", msg.strip())

    def create_main_frame(self):
        self.main_frame = QWidget()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        
        # add_axes 
        self.axes = self.fig.add_axes([0,0,1,1])

        # Bind the 'pick' event for clicking on one of the bars
        #
        self.canvas.mpl_connect('pick_event', self.on_pick)
        
        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        
        # Other GUI controls
        textbox_label = QLabel('N points:')
        self.textbox = QLineEdit()
        self.textbox.setMinimumWidth(50)
        pos_int = QRegExpValidator(QRegExp("\\d*"))
        self.textbox.setValidator(pos_int)
        self.textbox.setText('50')
        self.connect(self.textbox, SIGNAL('editingFinished ()'), self.on_draw)

        self.b1 = QRadioButton("Landmarks")
        self.b1.setChecked(True)
        self.b1.toggled.connect(lambda:self.btnstate(self.b1))
        self.b2 = QRadioButton("Spline")
        self.b2.toggled.connect(lambda:self.btnstate(self.b2))
        
        slider_label = QLabel('Adjust:')
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 10)
        self.slider.setValue(0)
        self.landmarksValue = 0;
        self.splineValue = 0;
        self.slider.setTracking(True)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.connect(self.slider, SIGNAL('valueChanged(int)'), self.on_draw)
        
        # Layout with box sizers
        hbox = QHBoxLayout()
        for w in [textbox_label, self.textbox, slider_label, self.b1, self.b2, self.slider]:
            hbox.addWidget(w)
            hbox.setAlignment(w, Qt.AlignVCenter)
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)
        
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

        #disable controls
        self.slider.setVisible(False)
        self.textbox.setVisible(False)

    def on_pick(self, event):
        if isinstance(event.artist, matplotlib.lines.Line2D):
            line = event.artist
            xdata, ydata = line.get_data()
            ind = event.ind
            self.selected.set_xdata(xdata[ind][0])
            self.selected.set_ydata(ydata[ind][0])
            self.selected.set_markersize(10)
            self.canvas.draw()
    
    def on_draw(self):
        self.currentNematode.nPoints = int(self.textbox.text())
        if self.b1.isChecked() == True:
            if self.landmarksValue == self.slider.value():
                self.statusBar().showMessage('Adjusted landmarks a maximum of %d pixels' % self.landmarksValue)
            else:
                self.landmarksValue = self.slider.value()
                self.currentNematode.adjustLandmarks(self.landmarksValue)
                self.currentNematode.adjustSpline(self.splineValue)
                self.statusBar().showMessage('Adjusted landmarks a maximum of %d pixels' % self.landmarksValue)
        else:
            if self.splineValue == self.slider.value():
                self.statusBar().showMessage('Adjusted spline a maximum of %d pixels' % self.splineValue)
            else:
                self.splineValue = self.slider.value()
                self.currentNematode.adjustLandmarks(self.landmarksValue)
                self.currentNematode.adjustSpline(self.splineValue)
                self.statusBar().showMessage('Adjusted spline a maximum of %d pixels' % self.splineValue)        
        self.currentNematode.updatePlot()
        self.canvas.draw()
            
    def btnstate(self,b):
      if b.text() == "Landmarks" and b.isChecked() == True:
            self.slider.setValue(self.landmarksValue)	
      if b.text() == "Spline" and b.isChecked() == True:
            self.slider.setValue(self.splineValue)
        
    def create_status_bar(self):
        self.status_text = QLabel(("Showing image "+str(self.currentNematodeIndex)+"/"+str(len(self.imagePaths))))
        self.statusBar().addWidget(self.status_text, 1)

    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(self, text, slot=None, shortcut=None, icon=None, tip=None, checkable=False, signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action

def main():
    app = QApplication(sys.argv)
    form = AppForm()
    form.show()
    app.exec_()

if __name__ == "__main__":
    main()
