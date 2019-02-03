import sys
import json
import os
import pickle
import numpy as np
import scipy.stats.mstats as stats
sys.path.insert(0,'../')
from .index import Index
from .stock_method import Methods
from importlib import import_module

stocksModule = {}

for asset in ['common_stock_build','preferred_stock_build']:
    stocksModule[asset] = import_module(''.join(['.',asset]),'assets')


class GBCE(Index):
    def __init__(self,debug):
        super().__init__('GBCE')
        self.debug = debug
        self.stocks = []
        self.tempStocks = []

        file = ''.join(['GBCE','.json'])

        if os.path.isfile(''.join(['./',file])):
            with open(''.join(['GBCE','.json']),'r') as json_file:
                print('Cargando Data')
                self.data = json.load(json_file)
        else:
            self.data = {'stocks':{}}
        self.selectStocksByMethod(method = None, debug = True)

    def saveData(self):

        for stock in self.stocks:
            if stock not in self.data['stocks']:
                self.data['stocks'][stock]={}

        with open(''.join(['GBCE','.json']),'w+') as json_file:

            json.dump(self.data,json_file)

    def selectStocksByMethod(self, method = None, debug = True):

        if debug:

            self.createStock(['Tea Corp','TEA','Casi Corporation',100, 0,0])
            self.createStockTrade(symbol='TEA',price=40,quantity=40,indicator=1)
            self.createStock(['Pop Corp','POP','Casi Corporation',100, 8,0])
            self.createStockTrade(symbol='POP',price=200,quantity=90,indicator=0)
            self.createStock(['Ale Corp','ALE','Casi Corporation',60, 23,0])
            self.createStockTrade(symbol='ALE',price=250,quantity=120,indicator=0)
            self.createStock(['Gin Corp','GIN','Casi Corporation',100, 8,1])
            self.createStockTrade(symbol='GIN',price=180,quantity=120,indicator=0)
            self.createStock(['Joe Corp','JOE','Casi Corporation',250, 13,0])
            self.createStockTrade(symbol='JOE',price=45,quantity=200,indicator=1)
        else:
            Methods(method)

    def createStock(self,stockData):
        name = stockData[0]
        symbol = stockData[1]
        description = stockData[2]
        pvalue = stockData[3]
        dividend = stockData[4]
        typo = stockData[5]
        if symbol not in self.stocks:
            self.stocks += [symbol]
        if symbol not in self.data['stocks']:
            self.data['stocks'][symbol]={}
            if typo == 0:
                stock = stocksModule['common_stock_build'].CommonStock(name,symbol,description,pvalue,dividend)
            else:
                stock = stocksModule['preferred_stock_build'].PreferredStock(name,symbol,description,pvalue,dividend)
            self.data['stocks'][symbol]['info'] = stock.getAssetInfo()
            self.tempStocks.append(stock)

        else:
            print('Stock Already Created')

    def createStockTrade(self,symbol,price,quantity,indicator):
        for idx,stock in enumerate(self.tempStocks):

            if symbol == stock.getAssetInfo()['Symbol']:
                currentStockIdx = idx
                break
        if 'currentStockIdx' in locals():
            self.tempStocks[currentStockIdx].getTradeInfo(quantity,price,indicator)
            if 'trade' not in self.data['stocks'][symbol]:
                self.data['stocks'][symbol]['trade'] = []
            self.data['stocks'][symbol]['trade'].append(self.tempStocks[currentStockIdx].getLastTrade().to_dict())
            self.data['stocks'][symbol]['trade'][-1]['Time'] = str(self.data['stocks'][symbol]['trade'][-1]['Time'])
            self.data['stocks'][symbol]['trade'][-1]['Quantity'] = int(self.data['stocks'][symbol]['trade'][-1]['Quantity'])
            self.data['stocks'][symbol]['trade'][-1]['Price'] = int(self.data['stocks'][symbol]['trade'][-1]['Price'])
            self.data['stocks'][symbol]['trade'][-1]['Indicator'] = bool(self.data['stocks'][symbol]['trade'][-1]['Indicator'])
            self.data['stocks'][symbol]['trade'][-1]['Amount'] = int(self.data['stocks'][symbol]['trade'][-1]['Amount'])

            self.saveData()
        else:
            print('Stock does not exist, create stock First')

    def getPERatio(self,symbol):
        for idx,stock in enumerate(self.tempStocks):

            if symbol == stock.getAssetInfo()['Symbol']:
                currentStockIdx = idx
                break
        if 'currentStockIdx' in locals():
            return self.tempStocks[currentStockIdx].getStockPE()
        else:

            print('Stock does not exist, create stock First')
            return None

    def getDividendYield(self,symbol):
        for idx,stock in enumerate(self.tempStocks):
            if symbol == stock.getAssetInfo()['Symbol']:
                currentStockIdx = idx
                break
        if 'currentStockIdx' in locals():
            return self.tempStocks[currentStockIdx].getDividendYield()
        else:

            print('Stock does not exist, create stock First')
            return None

    def loadFixedDividend(self,symbol,fixed):
        for idx,stock in enumerate(self.tempStocks):
            if symbol == stock.getAssetInfo()['Symbol']:
                currentStockIdx = idx
                break
        if 'currentStockIdx' in locals():
            self.tempStocks[currentStockIdx].getFixedDividend(fixed)

    def getWeightedPriceGeometricMean(self):
        vwsPrice = []
        for idx,stock in enumerate(self.tempStocks):
            vwsPrice+=[stock.calculateVolumeWeightedStockPrice()]
        return stats.gmean(np.array(vwsPrice))


    def loadVWSP(self,symbol,N):
        for idx,stock in enumerate(self.tempStocks):
            if symbol == stock.getAssetInfo()['Symbol']:
                currentStockIdx = idx
                break
        if 'currentStockIdx' in locals():
            return self.tempStocks[currentStockIdx].calculateVolumeWeightedStockPrice(N)
        else:
            print('Stock does not exist, create stock First')
            return None














