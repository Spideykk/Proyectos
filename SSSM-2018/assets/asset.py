from abc import ABC, abstractmethod, abstractproperty

class Asset(ABC):
	def __init__(self,name = None,symbol = None,description = None,pvalue = None,dividend = None):
		self.name = name
		self.symbol = symbol
		self.description = description
		self.dividend = dividend
		self.pvalue = pvalue

	def modifyDividend(self,dividend):

		return dividend

	def modifyParValue(self,pvalue):
		try:
			1/pvalue
		except ZeroDivisionError:
			print('Par Value can not be set to zero')
		else:
			return pvalue

	@abstractmethod
	def getAssetInfo(self):
		pass

	@abstractmethod
	def getTradeInfo(self):
		pass

	@abstractmethod
	def recordInfo(self):
		pass












