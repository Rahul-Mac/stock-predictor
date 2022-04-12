#!/usr/bin/env python3

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
import stock_predictor
          
if __name__ == "__main__":
    app = wx.App() 
    stock_predictor.stock_predictor(None) 
    app.MainLoop()  
