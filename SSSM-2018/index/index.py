import scipy as sc
from abc import ABC, abstractmethod, abstractproperty


class Index(ABC):
    def __init__(self,index):
        self.index = index

    def getGeometricMeanPrice(self):
        stocks = self.getStocksByIndex()
        prices = self.getPricesFromStocks(stocks)
        return sc.stats.mstats.gmean(prices)


    def getStocksByIndex(self):
        pass

    @abstractmethod
    def saveData(self,stocks):
        pass

    @abstractmethod
    def selectStocksByMethod(self):
        pass


