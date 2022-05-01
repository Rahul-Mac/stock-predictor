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

__author__ = "Rahul Mac"

import wx
import wx.lib.agw.ribbon as RB
import wx.grid as grid
import csv
import wx.adv
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
import locale
import manual

# Creates new IDs
INS = wx.Window.NewControlId()
DEL = wx.Window.NewControlId()
LIN = wx.Window.NewControlId()
LAS = wx.Window.NewControlId()
KNN = wx.Window.NewControlId()
SVM = wx.Window.NewControlId()
ABT = wx.Window.NewControlId()
MAN = wx.Window.NewControlId()

class stock_predictor(wx.Frame):
    def __init__(self, *args, **kw):
        super(stock_predictor, self).__init__(*args, **kw)
        locale.setlocale(locale.LC_ALL, 'C')
        self.SetTitle('Stock Predictor')
        self.ribbon = RB.RibbonBar(self, -1)
        self.icons()
        self.home()
        self.help()
        self.ribbon.Realize()
        self.mygrid = grid.Grid(self)
        s = wx.BoxSizer(wx.VERTICAL)
        s.Add(self.ribbon, 0, wx.EXPAND)
        s.Add(self.mygrid, 1, wx.EXPAND)
        self.SetSizer(s)
        self.statusBar = self.CreateStatusBar(style = wx.BORDER_NONE)
        self.statusBar.SetStatusText("Welcome to Stock Predictor")
        self.Show(True)
        self.Maximize(True)
        self.mygrid.CreateGrid(0, 0)
        self.data = []

    # Converts images in Bitmap format.
    # This function also sets the icon of the software.
    def icons(self):
        self.ins_bmp = wx.Bitmap('ins.png')
        self.del_bmp = wx.Bitmap('del.png')
        self.lin_bmp = wx.Bitmap('lin.png')
        self.las_bmp = wx.Bitmap('las.png')
        self.knn_bmp = wx.Bitmap('knn.png')
        self.svm_bmp = wx.Bitmap('svm.png')
        self.abt_bmp = wx.Bitmap('abt.png')
        self.man_bmp = wx.Bitmap('man.png')
        self.mla_bmp = wx.Bitmap('mla.png')
        self.dat_bmp = wx.Bitmap('dat.ico')
        self.SetIcon(wx.Icon("icon.ico"))

    # This function creates the "Home" Ribbon Page
    def home(self):
        self.home_page = RB.RibbonPage(self.ribbon, wx.ID_ANY, "Home")
        
        self.data_panel = RB.RibbonPanel(self.home_page, wx.ID_ANY, "Data", self.dat_bmp)
        self.data_button_bar = RB.RibbonButtonBar(self.data_panel)
        self.data_button_bar.AddSimpleButton(INS, "Load Data", self.ins_bmp, '')
        self.data_button_bar.AddSimpleButton(DEL, "Remove Data", self.del_bmp, '')
        self.data_button_bar.Bind(RB.EVT_RIBBONBUTTONBAR_CLICKED, self.load_data, id = INS)
        self.data_button_bar.Bind(RB.EVT_RIBBONBUTTONBAR_CLICKED, self.remove_data, id = DEL)
        self.data_button_bar.EnableButton(DEL, False)

        self.model_panel = RB.RibbonPanel(self.home_page, wx.ID_ANY, "Models", self.mla_bmp)
        self.model_button_bar = RB.RibbonButtonBar(self.model_panel)
        self.model_button_bar.AddSimpleButton(LIN, "Linear Regression", self.lin_bmp, '')
        self.model_button_bar.AddSimpleButton(LAS, "Lasso Regression", self.las_bmp, '')
        self.model_button_bar.AddSimpleButton(KNN, "K-Nearest Neighbors", self.knn_bmp, '')
        self.model_button_bar.AddSimpleButton(SVM, "Support Vector Machines", self.svm_bmp, '')
        self.model_button_bar.Bind(RB.EVT_RIBBONBUTTONBAR_CLICKED, self.lin_fun, id = LIN)
        self.model_button_bar.Bind(RB.EVT_RIBBONBUTTONBAR_CLICKED, self.las_fun, id = LAS)
        self.model_button_bar.Bind(RB.EVT_RIBBONBUTTONBAR_CLICKED, self.knn_fun, id = KNN)
        self.model_button_bar.Bind(RB.EVT_RIBBONBUTTONBAR_CLICKED, self.svm_fun, id = SVM)
        self.model_button_bar.EnableButton(LIN, False)
        self.model_button_bar.EnableButton(LAS, False)
        self.model_button_bar.EnableButton(KNN, False)
        self.model_button_bar.EnableButton(SVM, False)

        self.acc_panel = RB.RibbonPanel(self.home_page, wx.ID_ANY, "Accuracy")
        self.acc = wx.StaticText(self.acc_panel, id = wx.ID_ANY, label ="00.00%", style = wx.ALIGN_CENTRE_HORIZONTAL)
        font = wx.Font(wx.FontInfo(20))
        self.acc.SetFont(font)
        self.acc.SetForegroundColour(wx.Colour(0, 0, 0))

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
        GLOBAL_VALUE.ACC = "{:.2f}".format(float(acc*100))
        self.acc.SetLabel("Linear Regression = "+str(GLOBAL_VALUE.ACC) + "%")
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
        GLOBAL_VALUE.ACC = "{:.2f}".format(float(acc*100))
        self.acc.SetLabel("Lasso Regression = "+str(GLOBAL_VALUE.ACC) + "%")
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
    # If today’s price is lesser than the 500th day's price, we will store 1 (buy signal)
    # else we will store -1 (sell signal).
    # The number of neighbors used is 10. Finally, we will provide
    # a chart of predicted returns vs actual returns.
    def k_nearest_neighbours(self):
        df = pd.read_csv(GLOBAL_VALUE.FILE)
        df = df.dropna()
        df['Open-Close']= df['open'] - df['close']
        df['High-Low']  = df['high'] - df['low']
        df = df.dropna()
        X = df[['Open-Close', 'High-Low']]
        y = np.where(df['close'].shift(-500)>df['close'] , 1, -1)
        t = int(0.8*len(df))
        X_train = X[:t]
        y_train = y[:t]
        X_test = X[t:]
        y_test = y[t:]
        model = KNeighborsClassifier(n_neighbors=100).fit(X_train, y_train)
        df['pred'] = model.predict(X)
        acc = accuracy_score(y, df['pred'])
        GLOBAL_VALUE.ACC = "{:.2f}".format(float(acc*100))
        self.acc.SetLabel("K-Nearest Neighbours = "+str(GLOBAL_VALUE.ACC) + "%")
        df['ret'] = df['close'].pct_change()
        df['str'] = df['ret']*df['pred'].shift(500)
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

    # The SVM function takes the same values for X and y as KNN
    # The algorithm of this function is also similar that of KNN's
    def support_vector_machine(self):
        df = pd.read_csv(GLOBAL_VALUE.FILE)
        df = df.dropna()
        df['Open-Close']= df['open'] - df['close']
        df['High-Low']  = df['high'] - df['low']
        df = df.dropna()
        X = df[['Open-Close', 'High-Low']]
        y = np.where(df['close'].shift(-500)>df['close'],1, -1)
        t = int(0.8*len(df))
        X_train = X[:t]
        y_train = y[:t]
        X_test = X[t:]
        y_test = y[t:]
        model = SVC().fit(X_train, y_train)
        df['pred'] = model.predict(X)
        acc = accuracy_score(y, df['pred'])
        GLOBAL_VALUE.ACC = "{:.2f}".format(float(acc*100))
        self.acc.SetLabel("Support Vector Machine = "+str(GLOBAL_VALUE.ACC) + "%")
        df['ret'] = df['close'].pct_change()
        df['str'] = df['ret']*df['pred'].shift(500)
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

    def lin_fun(self, e):
        try:
            self.linear_regression()
        except Exception as e:
            wx.MessageBox(str(e), 'Error', wx.OK | wx.ICON_INFORMATION)

    def las_fun(self, e):
        try:
            self.lasso_regression()
        except Exception as e:
            wx.MessageBox(str(e), 'Error', wx.OK | wx.ICON_INFORMATION)

    def knn_fun(self, e):
        try:
            self.k_nearest_neighbours()
        except Exception as e:
            wx.MessageBox(str(e), 'Error', wx.OK | wx.ICON_INFORMATION)

    def svm_fun(self, e):
        try:
            self.support_vector_machine()
        except Exception as e:
            wx.MessageBox(str(e), 'Error', wx.OK | wx.ICON_INFORMATION)

    # Loads stock price file data (CSV)
    def load_data(self, e):
        openFileDialog = wx.FileDialog(self, "Open", "", "", "CSV files (*.csv)|*.csv", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        openFileDialog.ShowModal()
        GLOBAL_VALUE.FILE = openFileDialog.GetPath()
        if GLOBAL_VALUE.FILE == "":
            wx.MessageBox("File not selected", 'Error', wx.OK | wx.ICON_INFORMATION)
        else:
            self.view_table()
            self.model_button_bar.EnableButton(LIN, True)
            self.model_button_bar.EnableButton(LAS, True)
            self.model_button_bar.EnableButton(KNN, True)
            self.model_button_bar.EnableButton(SVM, True)
            self.data_button_bar.EnableButton(INS, False)
            self.data_button_bar.EnableButton(DEL, True)

    # Removes stock price data from table
    def remove_data(self, e):
        self.mygrid.DeleteRows(0, len(self.data))
        self.mygrid.DeleteCols(0, len(self.data[0]))
        self.data.clear()
        self.model_button_bar.EnableButton(LIN, False)
        self.model_button_bar.EnableButton(LAS, False)
        self.model_button_bar.EnableButton(KNN, False)
        self.model_button_bar.EnableButton(SVM, False)
        self.data_button_bar.EnableButton(INS, True)
        self.data_button_bar.EnableButton(DEL, False)
        self.acc.SetLabel("00.00%")

    # Displays information about the software
    def show_about(self, e):
        description = """
        Stock Predictor is a stock price prediction software that uses machine learning algorithms
        like Linear Regression, Lasso Regression, K-Nearest Neighbours, and Support Vector Machine
        to predict stock prices.
        """
        
        licence = """
        Stock Predictor
        Copyright (C) 2022  Rahul Mac
        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
        
        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.
        """
        
        info = wx.adv.AboutDialogInfo()
        info.SetIcon(wx.Icon('icon.ico'))
        info.SetName('Stock Predictor')
        info.SetVersion('2.0.0')
        info.SetDescription(description)
        info.SetCopyright('Copyright (C) 2022 Rahul Mac under GNU GPL v3 License')
        info.SetWebSite('https://github.com/Rahul-Mac/stock-predictor')
        info.SetLicence(licence)
        info.AddDeveloper('Rahul Mac')
        wx.adv.AboutBox(info)

    # Displays a manual 
    def show_manual(self, event):
        self.man = manual.manual(None, "Manual")
        self.man.ShowModal()
        self.man.Destroy()

    # This function creates the "Help" Ribbon Page
    def help(self):
        self.help_page = RB.RibbonPage(self.ribbon, wx.ID_ANY, "Help")
        
        self.info_panel = RB.RibbonPanel(self.help_page, wx.ID_ANY, "Information")
        self.info_button_bar = RB.RibbonButtonBar(self.info_panel)
        self.info_button_bar.AddSimpleButton(ABT, "About", self.abt_bmp, '')
        self.info_button_bar.AddSimpleButton(MAN, "Manual", self.man_bmp, '')
        self.info_button_bar.Bind(RB.EVT_RIBBONBUTTONBAR_CLICKED, self.show_about, id = ABT)
        self.info_button_bar.Bind(RB.EVT_RIBBONBUTTONBAR_CLICKED, self.show_manual, id = MAN)

    # Populates stock price data into the table
    def view_table(self):
        self.data.clear()
        try:
            with open(GLOBAL_VALUE.FILE, 'r') as stream:
                for rowdata in csv.reader(stream):
                    self.data.append(rowdata)
        except Exception as e:
            wx.MessageBox(str(e), 'Error', wx.OK | wx.ICON_INFORMATION)
        labels = self.data[0]
        del self.data[0]
        rows = len(self.data)
        cols = len(self.data[0])
        for i in range(len(labels)):
            self.mygrid.AppendCols(1)
            self.mygrid.SetColLabelValue( i, labels[i])
        for row in range (rows):
            self.mygrid.AppendRows(1)
            for col in range(cols):
                self.mygrid.SetCellValue(row, col, self.data[row][col])


