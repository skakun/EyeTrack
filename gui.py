import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize
from PyQt5 import Qt
import json
import os

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(1280, 720))
        self.setWindowTitle("Sentinel")
        self.alarm = False
        self.inform = False

        alarm_button = QPushButton('Zaalarmuj personel', self)
        inform_button = QPushButton('Powiadom personel', self)
        alarm_button.clicked.connect(self.on_alarm_click)
        inform_button.clicked.connect(self.on_inform_click)
        font = alarm_button.font()
        font.setPointSize(24)
        alarm_button.setFont(font)
        alarm_button.setStyleSheet("QPushButton {background:#ffbfbc; color: #3d3c3c} QPushButton:hover {background:#ffa499}")
        alarm_button.resize(640, 720)
        alarm_button.move(0, 0)

        inform_button.setFont(font)
        inform_button.setStyleSheet("QPushButton {background:#cceeff; color: #3d3c3c} QPushButton:hover {background: #9dd4f9}")
        inform_button.resize(640, 720)
        inform_button.move(640, 0)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            raise KeyboardInterrupt

    def on_alarm_click(self):
        print('Alarm')
        if not self.alarm:
            self.alarm = True
        else:
            self.alarm = False

        with open('alert_data.pkl', 'w') as f:
            f.truncate(0)
            alert_data = {'alarm': self.alarm, 'inform': self.inform}
            json.dumps(alert_data, f)

    def on_inform_click(self):
        print('Powiadomienie')
        if not self.inform:
            self.inform = True
        else:
            self.inform = False

        with open('alert_data.pkl', 'w') as f:
            f.truncate(0)
            alert_data = {'alarm': self.alarm, 'inform': self.inform}
            json.dumps(alert_data, f)


def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
