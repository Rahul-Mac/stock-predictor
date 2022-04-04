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

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
import stock_predictor
          
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = stock_predictor.stock_predictor()  # calls the main window
    app.exec_()
