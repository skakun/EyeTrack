import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(1280, 720))
        self.setWindowTitle("Sentinel")

        alarm_button = QPushButton('Zaalarmuj personel', self)
        inform_button = QPushButton('Powiadom personel', self)
        alarm_button.clicked.connect(self.on_alarm_click)
        inform_button.clicked.connect(self.on_inform_click)
        font = alarm_button.font()
        font.setPointSize(24)
        alarm_button.setFont(font)
        alarm_button.setStyleSheet("background-color: #ffa499")
        alarm_button.resize(640, 360)
        alarm_button.move(0, 0)

        inform_button.setFont(font)
        inform_button.setStyleSheet("background-color: #9dd4f9")
        inform_button.resize(640, 360)
        inform_button.move(640, 0)

    def on_alarm_click(self):
        print('Alarm')

    def on_inform_click(self):
        print('Powiadomienie')


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit( app.exec_() )
