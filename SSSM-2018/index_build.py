from asset import Asset
import numpy as np
from abc import ABC, abstractmethod, abstractproperty


class Index():
    def __init__(self,index):
        self.index = index

    def getGeometricMeanPrice(self):
        stocks = self.getStocksByIndex()
        prices = self.getPricesFromStocks(stocks)
        return np.sqrt(prices)

    @abstractmethod
    def getStocksByIndex(self):
        pass

    @abstractmethod
    def getPricesFromStocks(self,stocks):
        pass


