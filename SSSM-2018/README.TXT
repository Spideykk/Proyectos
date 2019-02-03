To use this package, please make sure python 3 is installed in your computer.
Before running the program, please delete always the GBCE.json file.
Then on CMD execute the following command on the SSSM-2018 folder:
	- pip3 install -r requirements.txt or python3 pip install requirements.txt
To execute the module execute in the SSSM-2018 package:
	- python3 main.py

The following commands are available on the program:
	help: Displays the help menu'
 	create SYM NAME DESCRIPTION PAR DIVIDEND TYPE FIXED(optional): creates a Stock, if symbol does not exist.
 	trade SYM PRICE QUANT INDICATOR: creates a trade for a stock, IF THE STOCK EXIST'
 	yield SYM: specifies the dividend yield of the stock'
 	vwsp SYM N: calculates the Volume Weighted Stock Price during N minutes'
	gmean: calculates the  geometric mean of the Volume Weighted Stock Price for all stocks'
	pe SYM: calculates the PE ratio of a stock'
	exit: exits the program

Example:
	trade SYM PRICE QUANT INDICATOR

All Trade Info will be  saved during each session in GBCE.json file.

Initial stocks Loaded.

symbol	TYPE 		Dividend    Par Value
	
TEA 	Common 		0 			100

POP 	Common 		8 			100

ALE 	Common 		23 			60

GIN 	Preferred 	8 	2% 		100

JOE 	Common 		13 			250