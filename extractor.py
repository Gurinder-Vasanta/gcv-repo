import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
from urllib.request import urlopen
import numpy as np
#data_dow = pd.read_html('https://markets.businessinsider.com/index/components/dow_jones')
stock_links=[]
stock_names=[]
stock_tickers=[]
stock_ids = []
size_counter = {}
ticker_size = {}

#tickers to remove:
#brk (index 15)
#bkng (index 21)
#bf (index 27)
#mmc (index 237)
for i in range(1,11):
    if(i==1):
        url='https://markets.businessinsider.com/index/components/s&p_500'
    else:
        url = 'https://markets.businessinsider.com/index/components/s&p_500?p='+str(i)
        data = requests.get(url).text
        soup = BeautifulSoup(data,'html.parser')
        table = soup.find('table')
        temp=str(table)
        table=temp.split('\n')
        for j in range(len(table)):
            if('<a' in table[j]):
                stock_links.append(table[j])
for i in range(len(stock_links)):
    dash_ind = stock_links[i].index('-')
    equal_ind = stock_links[i].index('=')
    wrap_ind = stock_links[i].index('>')
    stock_tickers.append(stock_links[i][17:dash_ind])
    stock_names.append(stock_links[i][dash_ind+equal_ind+equal_ind+1:wrap_ind-1])
stock_tickers = np.array(stock_tickers).reshape(len(stock_tickers),1)
stock_tickers = list(stock_tickers.reshape(len(stock_tickers)))
bad_indices = [15,28,83,237,181,100]
for i in range(len(bad_indices)):
    print(stock_tickers[bad_indices[i]])
    del(stock_tickers[bad_indices[i]])
    del(stock_links[bad_indices[i]])
    del(stock_names[bad_indices[i]])
#del(stock_tickers[15])
#del(stock_tickers[21])
#del(stock_tickers[28])
#del(stock_tickers[237])
#input('stop')
#smallest usable size will be 1962
#v len(stock_tickers)
ticker_size_file = open('ticker_size.csv','w')
ticker_size_file.write('Ticker,Size\n')
for i in range(len(stock_tickers)):
    print(stock_tickers[i][0])
    print(stock_tickers[i])
    print(i)
    temp = requests.get('https://markets.businessinsider.com/stocks/'+str(stock_tickers[i])+'-stock').text
    soup = BeautifulSoup(temp, 'html.parser')
    table = list(soup.find_all('script'))
    master_ind = 0
    for j in range(len(table)):
        table[j] = str(table[j])
        if('TKData' in table[j]):
            master_ind = j
    values = str(table[master_ind])
    dictionary = values.replace(', "StockMarkets"', '}')
    ind = dictionary.index('{')
    rind = dictionary.index('}')
    tkdata = (dictionary[ind:rind + 1])
    tkdata = tkdata.replace(' "ChartData" : null, ', ' ')
    b = tkdata.split(', ')
    tkdata = json.loads(tkdata)
    stock_ids.append(tkdata['TKData'])
    print(tkdata['TKData'])
    data = np.array(requests.get(
        'https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Share&tkData=' + tkdata[
            'TKData'] + '&from=19700201&to=20220524').json())
    t = data.shape
    ticker_size_file.write(str(stock_tickers[i])+','+str(t[0])+'\n')
    if(t[0] not in size_counter):
        size_counter[t[0]] = 1
    else:
        size_counter[t[0]] += 1
    ticker_size[stock_tickers[i]] = t[0]
    print(data.shape)
    print(size_counter)
#print(np.array(stock_links))
print(stock_ids)
tick_file = open('Tickers.txt','w')
tick_file.write(str(stock_tickers))
print(np.array(stock_tickers))
#print(np.array(stock_names))
count_files = open('Size_Counts.txt','w')
#count_files.write('ticker_size array below\n')
#count_files.write(str(ticker_size))
#print(ticker_size)

count_files = open('Size_Counts.txt','w')
count_files.write(str(size_counter))

#input('------------------------')
for i in range(len(stock_ids)):
    data = np.array(requests.get(
        'https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Share&tkData=' + str(stock_ids[i]) + '&from=19700201&to=20220515').json())
    output_file_name = stock_tickers[i]+'.csv'
    print(output_file_name)

    output_file = open('/Users/gcvasanta/Desktop/Python/Stock_Predictor/Data/'+output_file_name,'w')
    output_file.write('Prices\n')
    for j in range(len(data)):
        output_file.write(str(data[j]['Close'])+'\n')
    print(data)
'''for i in range(2,12):
    data_sp = pd.read_html('https://markets.businessinsider.com/index/components/s&p_500'+'?p='+str(i))
    print(data_sp)'''

#https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Index&tkData=1059,998434,1059,333&from=19700201&to=20220518
#above is link to sp500 spot price (starts at 4/27/1992)
#https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Index&tkData=310,998313,310,333&from=19700201&to=20220518
#above is link to dow jones spot price (starts at 4/27/1992)
#https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Index&tkData=1135,985336,1135,333&from=19700201&to=20220518
#above is link to nasdaq spot price (starts at 12/21/1998)

#{4601: 1, 7571: 5, 7331: 6, 1965: 91, 7330: 24, 7094: 1, 7574: 20, 7575: 86, 5692: 1, 7235: 1, 6268: 1, 3814: 1, 1926: 1, 6918: 1, 6970: 1, 6361: 1, 544: 2, 1962: 2, 4514: 1, 4359: 1, 4677: 1, 4219: 1, 5507: 1, 1509: 1, 4104: 1, 1924: 1, 7576: 13, 7573: 2, 5685: 1, 750: 1, 1866: 1, 1610: 1, 7572: 1, 6647: 1, 7577: 1, 6788: 1, 6675: 1, 3792: 1, 1964: 3, 4416: 1, 3757: 2, 3146: 1, 4493: 1, 796: 1, 7205: 1, 1183: 1, 1301: 1, 7143: 1, 2379: 1, 5570: 1, 1840: 1, 7241: 1, 7035: 1, 6666: 2, 1784: 1, 6536: 1, 994: 1, 665: 1, 4471: 1, 5259: 1, 2877: 1, 2873: 1, 1492: 1, 2683: 1, 801: 2, 6743: 1, 2992: 1, 5954: 1, 3086: 1, 2891: 1, 5366: 1, 5796: 1, 6300: 1, 2815: 1, 1655: 1, 2120: 1, 256: 1, 386: 1, 2807: 1, 7326: 1, 1261: 1, 2147: 1, 7332: 2, 3637: 1, 2271: 1, 1843: 1, 1886: 1, 5761: 1, 7098: 1, 1906: 1, 7329: 2, 2833: 1, 5544: 1, 1394: 1, 4384: 1, 3923: 1, 890: 1, 4116: 1, 3034: 1, 6029: 1, 2742: 1, 7109: 1, 4021: 1, 5781: 1, 6909: 1, 5563: 1, 6162: 1, 7122: 1, 865: 1, 6142: 1, 4751: 1, 4346: 1, 6015: 1, 7265: 1, 3649: 1, 863: 1, 1689: 1, 2347: 1, 4567: 1, 7189: 1, 2765: 1, 7578: 1, 253: 1, 5610: 1, 2036: 1, 1729: 2, 2003: 1, 1107: 1, 3567: 1, 2540: 1, 6232: 1, 1109: 1, 6158: 1, 5140: 1, 6664: 1, 1855: 1, 6103: 1, 6376: 1, 6273: 1, 6942: 1, 881: 1, 4500: 1, 6007: 1, 5693: 1, 615: 1, 6425: 1, 7313: 1, 4506: 1, 1342: 1, 250: 1, 6634: 1, 2485: 1, 6482: 1, 6856: 1, 790: 1, 6243: 1, 5436: 1, 5655: 1, 7059: 1, 872: 1, 7333: 1, 1649: 1, 4071: 1, 6289: 1, 2144: 1, 7562: 1, 6183: 1, 5907: 1, 1548: 1, 3878: 1, 928: 1, 5664: 1, 6140: 1, 378: 1, 3565: 1, 7307: 1, 7328: 1, 6774: 1, 1856: 1, 5852: 1, 1736: 1, 1602: 1, 1100: 1, 2664: 1, 6205: 1}
#{1965: 279, 7575: 2, 7574: 8, 206: 1, 7329: 6, 7333: 5, 3215: 1, 4177: 1, 7331: 6, 1926: 1, 543: 1, 1962: 2, 2237: 1, 7332: 18, 1509: 1, 1950: 1, 1924: 1, 6654: 1, 4893: 1, 6017: 1, 7577: 4, 7092: 1, 750: 1, 7187: 1, 1865: 1, 7284: 1, 4301: 1, 2412: 1, 6765: 1, 6847: 1, 796: 1, 1183: 1, 1296: 1, 5950: 1, 2548: 1, 1840: 1, 1784: 1, 994: 1, 7570: 1, 4238: 1, 5776: 1, 1964: 4, 3143: 1, 1487: 1, 802: 1, 801: 1, 110: 1, 7576: 1, 6673: 1, 1655: 1, 7573: 1, 5485: 1, 6623: 1, 1261: 1, 5517: 1, 3881: 1, 1843: 1, 1885: 1, 1906: 1, 1390: 1, 890: 1, 4686: 1, 4411: 1, 6077: 1, 2514: 1, 865: 1, 5264: 1, 4403: 1, 6471: 1, 4345: 1, 6664: 1, 5030: 1, 2243: 2, 1689: 1, 1108: 1, 5867: 1, 2964: 1, 2863: 1, 253: 1, 544: 1, 1729: 2, 7044: 1, 4031: 1, 5176: 1, 4086: 1, 1855: 1, 7185: 1, 4139: 1, 1342: 1, 2987: 1, 6391: 1, 790: 1, 4274: 1, 5381: 1, 5936: 1, 2991: 1, 2277: 1, 7111: 1, 3664: 1, 1539: 1, 4099: 1, 6114: 1, 3173: 1, 1856: 1, 1735: 1, 5265: 1, 4922: 1, 7560: 1}
#{1965: 186, 7574: 4, 7573: 34, 206: 1, 7329: 5, 7333: 5, 3215: 1, 4177: 1, 7331: 6, 1926: 1, 543: 1, 1962: 2, 2237: 1, 7332: 10, 1509: 1, 1950: 1, 1924: 1, 6654: 1, 4893: 1, 6017: 1, 7576: 4, 7092: 1, 750: 1, 7187: 1, 1865: 1, 7284: 1, 4301: 1, 2412: 1, 6765: 1, 6847: 1, 796: 1, 1183: 1, 1296: 1, 5950: 1, 2548: 1, 1840: 1, 1784: 1, 994: 1, 7569: 4, 4238: 1, 5776: 1, 1964: 4, 3143: 1, 1487: 1, 802: 1, 801: 1, 110: 1, 7575: 1, 6673: 1, 1655: 1, 7572: 6, 5485: 1, 6623: 1, 1261: 1, 5517: 1, 3881: 1, 1843: 1, 1886: 1, 7098: 1, 1906: 1, 2833: 1, 1390: 1, 4384: 1, 3923: 1, 890: 1, 4116: 1, 3034: 1, 2742: 1, 7109: 1, 4021: 1, 6909: 1,  2514: 1, 6162: 1, 7122: 1, 865: 1, 6142: 1, 4751: 1, 5264: 1, 6015: 1, 3649: 1, 5030: 1, 863: 1, 1689: 1, 7330: 9, 2347: 1, 7189: 1, 2964: 1, 2863: 1, 2765: 1, 253: 1, 544: 1, 5610: 1, 2036: 1, 1729: 2, 2003: 1, 1107: 1, 3567: 1, 2540: 1, 6232: 1, 1109: 1, 6158: 1, 5140: 1, 6664: 1, 1855: 1, 6103: 1, 6376: 1, 6273: 1, 6942: 1, 881: 1, 4500: 1, 5693: 1, 615: 1, 6425: 1, 7313: 1, 4506: 1, 1342: 1, 250: 1, 6634: 1, 2485: 1, 6482: 1, 6391: 1, 6856: 1, 790: 1, 6243: 1, 5936: 1, 5436: 1, 3757: 1, 5655: 1, 7059: 1, 2991: 1, 1649: 1, 4071: 1, 6289: 1, 7560: 1, 6183: 1, 1548: 1, 3878: 1, 928: 1, 6140: 1}
#{4601: 1, 7568: 2, 7331: 6, 1965: 92, 7330: 24, 7094: 1, 7571: 61, 7572: 42, 5692: 1, 7235: 1, 6268: 1, 3814: 1, 1926: 1, 6918: 1, 6970: 1, 6361: 1, 544: 2, 1962: 2, 4514: 1, 4359: 1, 4677: 1, 4219: 1, 5507: 1, 1509: 1, 4104: 1, 1924: 1, 7570: 12, 7573: 3, 7569: 3, 5685: 1, 750: 1, 1866: 1, 1610: 1, 6647: 1, 7574: 2, 6788: 1, 6675: 1, 3792: 1, 1964: 3, 4416: 1, 3757: 2, 3146: 1, 4493: 1, 796: 1, 7205: 1, 1183: 1, 1301: 1, 7143: 1, 2379: 1, 5570: 1, 1840: 1, 7241: 1, 7035: 1, 6666: 2, 1784: 1, 6536: 1, 994: 1, 665: 1, 4471: 1, 5259: 1, 2877: 1, 2873: 1, 1492: 1, 2683: 1, 801: 2, 6743: 1, 2992: 1, 5954: 1, 3086: 1, 2891: 1, 5366: 1, 5796: 1, 6300: 1, 2815: 1, 1655: 1, 2120: 1, 256: 1, 386: 1, 2807: 1, 7326: 1, 1261: 1, 2147: 1, 7332: 2, 3637: 1, 2271: 1, 1843: 1, 1886: 1, 5761: 1, 7098: 1, 1906: 1, 7329: 2, 2833: 1, 5544: 1, 1394: 1, 4384: 1, 3923: 1, 890: 1, 4116: 1, 3034: 1, 6029: 1, 2742: 1, 7109: 1, 4021: 1, 5781: 1, 6909: 1, 5563: 1, 6162: 1, 7122: 1, 865: 1, 6142: 1, 4751: 1, 4346: 1, 6015: 1, 7265: 1, 3649: 1, 863: 1, 1689: 1, 2347: 1, 4567: 1, 7189: 1, 2765: 1, 253: 1, 5610: 1, 2036: 1, 1729: 2, 2003: 1, 1107: 1, 3567: 1, 2540: 1, 6232: 1, 7567: 3, 1109: 1, 6158: 1, 5140: 1, 6664: 1, 1855: 1, 6103: 1, 6376: 1, 6273: 1, 6942: 1, 881: 1, 4500: 1, 6007: 1, 5693: 1, 615: 1, 6425: 1, 7313: 1, 4506: 1, 1342: 1, 250: 1, 6634: 1, 2485: 1, 6482: 1, 6856: 1, 790: 1, 6243: 1, 5436: 1, 5655: 1, 7059: 1, 872: 1, 7333: 1, 1649: 1, 4071: 1, 6289: 1, 2144: 1, 7558: 1, 6183: 1, 5907: 1, 1548: 1, 3878: 1, 928: 1, 5664: 1, 6140: 1, 378: 1, 3565: 1, 7307: 1, 7328: 1, 6774: 1, 1856: 1, 5852: 1, 1736: 1, 1602: 1, 1100: 1, 2664: 1, 6205: 1}
