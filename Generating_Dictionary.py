import requests
textfile = open('TickerID_Ticker_Symbol_Dictionary.txt',"w")
for b in range(65, 91):
    print(chr(b))
    temp = requests.get('https://quotes-gw.webullfintech.com/api/search/pc/tickers?keyword=' + chr(b) + '&pageIndex=1&pageSize=1000').json()
    a = (list(temp['data']))
    for i in range(len(a)):
        if(a[i]['disExchangeCode'] == 'NYSE' or a[i]['disExchangeCode'] == 'NASDAQ'):
            textfile.write('Ticker ID: ' + str(a[i]['tickerId']) + ' Company Name: ' + str(a[i]['name']) + ' Company Symbol: ' + str(a[i]['disSymbol']) + "\n")
