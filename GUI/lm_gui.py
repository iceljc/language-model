import sys
import os
import subprocess
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit, QTextEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize    

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(500, 500))    
        self.setWindowTitle("testgui") 

        

        self.nameLabel = QLabel(self)
        self.nameLabel.setText('Input text:')
        self.line = QTextEdit(self)

        self.line.move(100, 20)
        self.line.resize(300, 100)
        self.nameLabel.move(20, 20)

        pybutton = QPushButton('OK', self)
        pybutton.clicked.connect(self.inputData)
        pybutton.resize(100,32)
        pybutton.move(120, 140)

        btn_clear = QPushButton('Cancel', self)
        btn_clear.clicked.connect(self.clearText)
        btn_clear.resize(100,32)
        btn_clear.move(220, 140)


        self.nameLabel2 = QLabel(self)
        self.nameLabel2.setText('Target word:')
        self.line2 = QTextEdit(self)

        self.line2.move(100, 220)
        self.line2.resize(300, 32)
        self.nameLabel2.move(20, 220)

        pybutton2 = QPushButton('OK', self)
        pybutton2.clicked.connect(self.appendData)
        pybutton2.resize(100,32)
        pybutton2.move(120, 260)

        btn_clear2 = QPushButton('Cancel', self)
        btn_clear2.clicked.connect(self.clearText2)
        btn_clear2.resize(100,32)
        btn_clear2.move(220, 260)

        self.nameLabel3 = QLabel(self)
        self.nameLabel3.setText('Prediction:')
        self.nameLabel3.move(20, 320)

        self.out = QLineEdit(self)
        self.out.move(100, 320)
        self.out.resize(300, 32)

        pybutton3 = QPushButton('Run', self)
        pybutton3.clicked.connect(self.outputData)
        pybutton3.resize(100,32)
        pybutton3.move(120, 360)



    def inputData(self):
        text = self.line.toPlainText()

        fid = open("/mnt/sdd/iceljc/pytorch-language-model/data/ptb.test.txt", "w+")
        fid.write(text)
        fid.close()

    def appendData(self):
        text = self.line2.toPlainText()
        if text != '':
            if (os.path.isfile("/mnt/sdd/iceljc/pytorch-language-model/data/ptb.test.txt")):
                if (os.stat("/mnt/sdd/iceljc/pytorch-language-model/data/ptb.test.txt").st_size != 0):
                    fid = open("/mnt/sdd/iceljc/pytorch-language-model/data/ptb.test.txt", "a+")
                    fid.write(' ' + text + ' ')
                    fid.close()
                else:
                    print("Empty file ! Please input text first !")
            else:
                print("Please input text first !")
    
    def outputData(self):
        res = "hello"
        run_cmd = ['python3', 'lm_model_load.py']
        result = subprocess.check_output(run_cmd, universal_newlines = True)
        self.out.setText(result)
        

    def clearText(self):
        self.line.clear()

    def clearText2(self):
        self.line2.clear()

    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


