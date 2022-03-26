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
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem, QFileDialog, QApplication
import csv
import GLOBAL_VALUE
import math
import pandas as pd
import numpy as np
import matplotlib. pyplot as plt
import seaborn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import sklearn.utils._typedefs
import sklearn.neighbors._partition_nodes
import manual

class stock_predictor(QtWidgets.QMainWindow):
    def __init__(self):
        super(stock_predictor, self).__init__()
        uic.loadUi('stock_predictor.ui', self)
        self.connections()
        self.measurements()
        self.names()
        self.setWindowIcon(QtGui.QIcon('icon.ico'))
        self.show()

    # This function is used to set texts
    # for window, buttons, and status bar
    def names(self):
        self.setWindowTitle("Stock Predictor")
        self.statusBar().showMessage("Welcome to Stock Predictor v1.0.0")
        self.lin_reg_btn.setText("Linear\nRegression")
        self.lasso_btn.setText("Lasso\nRegression")
        self.knn_btn.setText("k-Nearest\nNeighbors")
        self.svm_btn.setText("Sprt. Vector\nMachine")

    # Connects button click to function
    def connections(self):
        self.load.clicked.connect(self.load_data)
        self.remove.clicked.connect(self.clear_table)
        self.lin_reg_btn.clicked.connect(self.lin_reg)
        self.lasso_btn.clicked.connect(self.lasso)
        self.knn_btn.clicked.connect(self.knn)
        self.svm_btn.clicked.connect(self.svm)
        self.abt.clicked.connect(self.abt_win)
        self.lic.clicked.connect(self.lic_win)
        self.man.clicked.connect(self.manual)

    def abt_win(self):
        text = "Stock Predictor v0.1.0\nis a stock price prediction software.\n\nCopyright (C) 2022 Rahul Mac\n under GNU GPL v3 License"
        QMessageBox().about(self, "About", text)

    def lic_win(self):
        text = "\t\t\tStock Predictor\n\
        Copyright (C) 2022  Rahul Mac\n\n\
        This program is free software: you can redistribute it and/or modify\n\
        it under the terms of the GNU General Public License as published by\n\
        the Free Software Foundation, either version 3 of the License, or\n\
        (at your option) any later version.\n\n\
        This program is distributed in the hope that it will be useful,\n\
        but WITHOUT ANY WARRANTY; without even the implied warranty of\n\
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n\
        GNU General Public License for more details."
        QMessageBox().about(self, "License", text)

    def manual(self):
        self.window = manual.manual()
        self.window.show()

    # function for position and size of various widgets
    def measurements(self):
        desktop = QApplication.desktop()
        screenRect = desktop.screenGeometry()
        width = screenRect.width()
        height = screenRect.height()
        self.ribbon.setGeometry(0, 0, width, 141)
        self.data_table.setGeometry(0, 145, width, height - 250)
        self.showMaximized()


    # function that performs linear regression on the given stock data file.
    # Here, X i.e. moving average, is the independent variable
    # and y i.e. the closing price, is the dependent variable.
    # The stock price will be predicted based on moving average.
    def linear_regression(self):
        df = pd.read_csv(GLOBAL_VALUE.FILE)
        df = df.dropna()
        df = df.filter(['close'])
        df['Avg']= df['close'].shift(1).rolling(window=10).mean()
        df= df.dropna()
        X = df[['Avg']]
        y = df[['close']]
        t = int(0.8*len(df))
        X_train = X[:t]
        y_train = y[:t]
        X_test = X[t:]
        y_test = y[t:]
        model = LinearRegression().fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = model.score(pred, y_test)
        GLOBAL_VALUE.ACC = acc*100
        pred = pd.DataFrame(pred,index = y_test.index, columns = ['price'])
        plt.plot(y)
        plt.plot(y_test)
        plt.plot(pred)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(['Closing Price', 'Test Value', 'Prediction'], loc = "upper left")
        plt.show()

    # function that performs lasso regression on the given stock data file.
    # This function is similar to linear regression. The only difference here
    # is that it uses lasso regression.
    def lasso_regression(self):
        df = pd.read_csv(GLOBAL_VALUE.FILE)
        df = df.dropna()
        df = df.filter(['close'])
        df['Avg']= df['close'].shift(1).rolling(window=10).mean()
        df= df.dropna()
        X = df[['Avg']]
        y = df[['close']]
        t = int(0.8*len(df))
        X_train = X[:t]
        y_train = y[:t]
        X_test = X[t:]
        y_test = y[t:]
        model = Lasso().fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = pred.reshape(-1, 1)
        acc = model.score(pred, y_test)
        GLOBAL_VALUE.ACC = acc*100
        pred = pd.DataFrame(pred,index = y_test.index, columns = ['price'])
        plt.plot(y)
        plt.plot(y_test)
        plt.plot(pred)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(['Closing Price', 'Test Value', 'Prediction'], loc = "upper left")
        plt.show()

    # Function for kNN
    # Here, the independent variable is ['Open-Close', 'High-Low']. 
    # The dependent variable stores the trading signal.
    # If tomorrow’s price is greater than today’s price then we will store 1 (buy signal)
    # and if it's lesser than today's price then we will store -1 (sell signal).
    # The number of neighbors used is 10. Finally, we will provide
    # a chart of predicted returns vs actual returns.
    def k_nearest_neighbors(self):
        df = pd.read_csv(GLOBAL_VALUE.FILE)
        df = df.dropna()
        df['Open-Close']= df['open'] - df['close']
        df['High-Low']  = df['high'] - df['low']
        df = df.dropna()
        X = df[['Open-Close', 'High-Low']]
        y = np.where(df['close'].shift(-1)>df['close'],1, -1)
        t = int(0.8*len(df))
        X_train = X[:t]
        y_train = y[:t]
        X_test = X[t:]
        y_test = y[t:]
        model = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
        df['pred'] = model.predict(X)
        acc = accuracy_score(y, df['pred'])
        GLOBAL_VALUE.ACC = acc*100
        df['ret'] = df['close'].pct_change()
        df['str'] = df['ret']*df['pred'].shift(1)
        df['ret'] = df['ret'].cumsum()
        df['str'] = df['str'].cumsum()
        df['ret'] = df['ret']*10
        df['str'] = df['str']*10
        plt.plot(df['ret'])
        plt.plot(df['str'])
        plt.xlabel('Date')
        plt.ylabel('Returns (%)')
        plt.legend(['Actual', 'Prediction'], loc = "upper left")
        plt.show()

    # The SVM function takes the same values for X and y as kNN 
    def support_vector_machine(self):
        df = pd.read_csv(GLOBAL_VALUE.FILE)
        df = df.dropna()
        df['Open-Close']= df['open'] - df['close']
        df['High-Low']  = df['high'] - df['low']
        df = df.dropna()
        X = df[['Open-Close', 'High-Low']]
        y = np.where(df['close'].shift(-1)>df['close'],1, -1)
        t = int(0.8*len(df))
        X_train = X[:t]
        y_train = y[:t]
        X_test = X[t:]
        y_test = y[t:]
        model = SVC().fit(X_train, y_train)
        df['pred'] = model.predict(X)
        acc = accuracy_score(y, df['pred'])
        GLOBAL_VALUE.ACC = acc*100
        df['ret'] = df['close'].pct_change()
        df['str'] = df['ret']*df['pred'].shift(1)
        df['ret'] = df['ret'].cumsum()
        df['str'] = df['str'].cumsum()
        df['ret'] = df['ret']*10
        df['str'] = df['str']*10
        plt.plot(df['ret'])
        plt.plot(df['str'])
        plt.xlabel('Date')
        plt.ylabel('Returns (%)')
        plt.legend(['Actual', 'Prediction'], loc = "upper left")
        plt.show()

    # function to load the stock file
    def load_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName,_ = QFileDialog.getOpenFileName(self,"Select Stock Data File", "","CSV Files (*.csv)", options=options)
        if fileName:
            self.view_table(fileName)
        else:
             QMessageBox.critical(self, "Error", "Error loading file! Try again")


    # The below give four functions lin_reg, lasso, knn, and svm call the ML functions 
    # and set text for accuracy and status bar.

    def lin_reg(self):
        if GLOBAL_VALUE.FILE == "":
            QMessageBox.critical(self, "Error", "Stock data file has not been loaded")
        else:
            try:
                self.linear_regression()
                GLOBAL_VALUE.ACC = "{:.2f}".format(float(GLOBAL_VALUE.ACC))
                self.acc.setText(str(GLOBAL_VALUE.ACC)+"%")
                self.statusBar().showMessage("Linear Regression Accuracy = "+str(GLOBAL_VALUE.ACC)+"%")
            except:
                QMessageBox.critical(self, "Error", "Graph plotting failed")

    def lasso(self):
        if GLOBAL_VALUE.FILE == "":
            QMessageBox.critical(self, "Error", "Stock data file has not been loaded")
        else:
            try:
                self.lasso_regression()
                GLOBAL_VALUE.ACC = "{:.2f}".format(float(GLOBAL_VALUE.ACC))
                self.acc.setText(str(GLOBAL_VALUE.ACC)+"%")
                self.statusBar().showMessage("Lasso Regression Accuracy = "+str(GLOBAL_VALUE.ACC)+"%")
            except:
                QMessageBox.critical(self, "Error", "Graph plotting failed")

    def knn(self):
        if GLOBAL_VALUE.FILE == "":
            QMessageBox.critical(self, "Error", "Stock data file has not been loaded")
        else:
            try:
                self.k_nearest_neighbors()
                GLOBAL_VALUE.ACC = "{:.2f}".format(float(GLOBAL_VALUE.ACC))
                self.acc.setText(str(GLOBAL_VALUE.ACC)+"%")
                self.statusBar().showMessage("k-Nearest Neighbors Accuracy = "+str(GLOBAL_VALUE.ACC)+"%")
            except:
                QMessageBox.critical(self, "Error", "Graph plotting failed")

    def svm(self):
        if GLOBAL_VALUE.FILE == "":
            QMessageBox.critical(self, "Error", "Stock data file has not been loaded")
        else:
            try:
                self.support_vector_machine()
                GLOBAL_VALUE.ACC = "{:.2f}".format(float(GLOBAL_VALUE.ACC))
                self.acc.setText(str(GLOBAL_VALUE.ACC)+"%")
                self.statusBar().showMessage("Support Vector Machine Accuracy = "+str(GLOBAL_VALUE.ACC)+"%")
            except:
                QMessageBox.critical(self, "Error", "Graph plotting failed")

    # Removes all the entries from the table and the columns.
    # It also resets the accuracy to 00.00%.
    def clear_table(self):
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(0)
        self.statusBar().showMessage("Table cleared")
        self.acc.setText("00.00%")

    # Displays the stock price data in the main table        
    def view_table(self, file):
        data = []
        with open(file, 'r') as stream:
            for rowdata in csv.reader(stream):
                data.append(rowdata)
        labels = data[0]
        del data[0]
        nb_row = len(data)
        nb_col = len(data[0])
        self.data_table.setRowCount(nb_row)
        self.data_table.setColumnCount(nb_col)
        self.data_table.setHorizontalHeaderLabels(labels)
        header = self.data_table.horizontalHeader() 
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        for row in range (nb_row):
            for col in range(nb_col):
                item = QTableWidgetItem(str(data[row][col]))
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.data_table.setItem(row, col, item)
        GLOBAL_VALUE.FILE = file
        self.statusBar().showMessage("Stock data loaded\tFile:"+GLOBAL_VALUE.FILE)

