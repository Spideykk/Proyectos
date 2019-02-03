from importlib import import_module
import json

stocksModule = import_module('assets.common_stock_build','SSM-2018')
indexModule = import_module('.GBCE_index','index')




testIndex = indexModule.GBCE(debug=True)
testIndex.createStock(['Casi Corp','CCO','Casi Corporation',105, 0,1])
testIndex.createStockTrade(symbol='CCO',price=0,quantity=40,indicator=1)
testIndex.createStockTrade(symbol='CCO',price=105,quantity=40,indicator=0)

testIndex.loadFixedDividend('CCO',0.2)
print(testIndex.getPERatio('CCO'))
print(testIndex.getDividendYield('CCO'))
print(testIndex.getWeightedPriceGeometricMean())
