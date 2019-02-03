import sys
import json
import os
import pickle

import numpy as np
import scipy.stats.mstats as stats
sys.path.insert(0,'../')
from importlib import import_module

indexModule = {}

for index in ['GBCE_index']:
    indexModule[index] = import_module(''.join(['.',index]),'index')

data = 'initilize'
print('Initializing Stock Trader')
index = indexModule['GBCE_index'].GBCE(debug=True)
while(data != 'exit'):

    print('System Loaded')
    data= input('Introduce command: ')
    if data == 'help':
        print('This is a Basic Stock Trader Visualizer. It has the following commands:')
        print('help: Displays the help menu')
        print('create SYM NAME DESCRIPTION PAR DIVIDEND TYPE FIXED(optional): creates a Stock, if symbol does not exist.')
        print('trade SYM PRICE QUANT INDICATOR: creates a trade for a stock, IF THE STOCK EXIST')
        print('yield SYM: specifies the dividend yield of the stock')
        print('vwsp SYM N: calculates the Volume Weighted Stock Price during N minutes')
        print('gmean: calculates the  geometric mean of the Volume Weighted Stock Price for all stocks')
        print('pe SYM: calculates the PE ratio of a stock')
        print('exit: exits the program')
    elif data.split()[0] == 'create':
        symbol = data.split()[1]
        name = data.split()[2]
        descr = data.split()[3]
        pvalue = abs(float(data.split()[4]))
        dividend = abs(float(data.split()[5]))
        typo = bool(data.split()[6])
        index.createStock([name,symbol,descr,pvalue, dividend,typo])
        if len(data.split()) == 8:
            fixed = float(data.split()[7])
            index.loadFixedDividend(symbol,fixed)
         print('Stock created')

    elif data.split()[0] == 'trade':
        try:
            symbol = data.split()[1]
            price = abs(float(data.split()[2]))
            quantity = abs(int(data.split()[3]))
            indicator = bool(data.split()[4])
            index.createStockTrade(symbol,price,quantity,indicator)
            print('Stock Trade created')
        except:
            print('execute command correctly')

    elif data.split()[0] == 'yield':
        try:
            symbol = data.split()[1]
            print('Dividend yield for %s is %s'% (symbol,index.getDividendYield(symbol)))
        except:
            print('execute command correctly')

    elif data.split()[0] == 'vwsp':
        try:
            symbol = data.split()[1]
            minutes = abs(data.split()[2])
            print(symbol)
            print('Volume Stock Price')
            print(index.calculateVolumeWeightedStockPrice(symbol,minutes))
        except:
            print('execute command correctly')

    elif data.split()[0] == 'pe':
        try:
            symbol = data.split()[1]
            print(symbol)
            print('PE ratio')
            print(index.getPERatio(symbol))
        except:
            print('execute command correctly')

    elif data.split()[0] == 'gmean':
        print('Index Geometric Mean')
        print(index.getWeightedPriceGeometricMean())
    elif data.split()[0] == 'exit':
        data= 'exit'
    else:
        print('Write a correct command')

print('Closing program transactions info saved in GBCE.json')

