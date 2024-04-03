import sys
import os
from amlyprotein import *
from PyQt5 import QtWidgets, QtGui, QtCore

class MyForm(QtWidgets.QMainWindow):
  def __init__(self,parent=None):
     QtWidgets.QWidget.__init__(self,parent)
     self.ui = Ui_MainWindow()
     self.ui.setupUi(self)
     self.ui.pushButton.clicked.connect(self.gbcacc)
     self.ui.pushButton_2.clicked.connect(self.nnacc)
     self.ui.pushButton_3.clicked.connect(self.nnpred)
     self.ui.pushButton_4.clicked.connect(self.gbcpred)

  def gbcacc(self):
    os.system("python -W ignore gbc1.py")

  def nnacc(self):
    os.system("python -W ignore nn1.py")

  def nnpred(self):
    os.system("python -W ignore nn2.py")

  def gbcpred(self):  
    os.system("python -W ignore gbc2.py")

       
if __name__ == "__main__":  
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())
