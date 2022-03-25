'''
This file is part of Stock Predictor.

Stock Predictor is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
Hardware Service Manager is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Stock Predictor.
If not, see <https://www.gnu.org/licenses/>.
'''

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication

class manual(QtWidgets.QDialog):
    def __init__(self):
        super(manual, self).__init__()
        uic.loadUi('manual.ui', self)
        self.setWindowTitle("Manual")
        self.setWindowIcon(QtGui.QIcon('icon.ico'))
        self.show()
