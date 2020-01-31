#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:50:32 2020

@author: rsehgal
"""

import pandas as pd

def GetData(filename,colNames):
    data=pd.read_csv(filename,delimiter=' ',names=colNames)
    return data