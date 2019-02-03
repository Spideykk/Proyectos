from .asset import Asset
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PreferredStock(Asset):
    def __init__(self,name,symbol,description,pvalue,dividend):
        super().__init__(name,symbol,description,pvalue,dividend)
        self.currentPrice = None
        self.type = 1    # Zero to identify common stock
        self.fixedDividend = None

    def getAssetInfo(self):
        return {'Name':self.name, 'Description':self.description, 'Par Value':self.pvalue, 'Dividend':self.fixedDividend,'Symbol':self.symbol}

    def getPrice(self,price):
        try:
            self.currentPrice = float(price)
            1/self.currentPrice
        except ValueError:
          print('Introduce a number to set price')
        except ZeroDivisionError:
          print('Price can not be Zero, use another value')

    def getFixedDividend(self,fixedDividend):
       try:
           self.fixedDividend = float(fixedDividend)
       except ValueError:
          print('Introduce a number to set fixed dividend')

    def getStockPE(self):
        try:
          return self.currentPrice/self.dividend
        except ZeroDivisionError:
          print ('PE set to Zero')
          return 0

    def getDividendYield(self):
        try:
            return self.fixedDividend*self.pvalue/self.currentPrice
        except ZeroDivisionError:
            print('Dividend is zero. There is no yield')

    def recordInfo(self,tradeInfo):
        if not hasattr(self,'state'):
            self.state = pd.DataFrame(tradeInfo,index=[1])

        else:
            self.state = self.state.append(pd.DataFrame(tradeInfo,index=[1]),ignore_index=True,sort=True)

    def getTradeInfo(self,quantity,price,indicator):
        trade = {'Time':datetime.now(),'Quantity':int(quantity), 'Price':price, 'Indicator':indicator,'Amount':quantity*price}
        try:
            1/price
            self.currentPrice = float(price)
        except ValueError:
          print('Introduce a number to set price')
        except ZeroDivisionError:
          print('Price can not be Zero, use another value')
        self.recordInfo(trade)


    def calculateVolumeWeightedStockPrice(self,minutes = 5):
        currentTime = datetime.now()
        deltaTime = timedelta(minutes=minutes)
        quantity = np.array(self.state['Quantity'][self.state['Time']>=currentTime-deltaTime].loc[:].values)
        price = np.array(self.state['Price'][self.state['Time']>=currentTime-deltaTime].loc[:].values)
        return np.average(price,weights=quantity)

    def getLastTrade(self):
        return self.state.iloc[-1]



