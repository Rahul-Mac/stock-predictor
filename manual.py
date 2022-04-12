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

MANUAL = """ 
Stock Predictor

Stock Predictor is a desktop application used for stock price prediction.


** Ribbon Bar **

The ribbon bar has two tabs - Home and Help.

== Home ==
-> Stock Data
	Load Data - Select stock file (CSV only).
	Remove Data - Clear the main table

-> Models
	Linear Regression
	Lasso Regression
	k-Nearest Neighbors
	Support Vector Machine

-> Accuracy
	Displays the accuracy of the selected model.

== Help ==
-> Information
	About - Brief information about the software
	Manual - The document that you are currently reading


** Table **
Information about the stock price


** Status Bar **
Information about the current activity

Written by Rahul Mac
"""
  
class manual(wx.Dialog): 
   def __init__(self, parent, title): 
      super(manual, self).__init__(parent, title = title, size = (700, 635)) 
      panel = wx.Panel(self) 
      text = wx.TextCtrl(panel, size = (700, 600), value = MANUAL, style = wx.TE_MULTILINE | wx.TE_READONLY)
