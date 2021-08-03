import requests
# v take the request url as user input
link = str(input("Enter the request URL: "))
# v take the company name as user input
company = str(input("Enter the company name: "))
# v take the name of the text file as user input
f_name = str(input("Enter the output file name: "))
# v pulls json file from website
temp = requests.get(link).json()
# v converts json to python dictionary
dictionary = dict(temp[0])
# v gets only the data from the python dictionary
AllData = dictionary["data"]
# v get the ticker id for apple
TickerId = dictionary["tickerId"]
# v python list to store prices
Price = []
# v nested for loops to store the price data in the Price python list
for i in range(len(AllData)):
    # v temporary variable to store the list that results from the split
    test = AllData[i].split(',')
    # v print statement to see which values are the 10 second by 10 second price
    print(test[1:5])
    # v for loop to append each price data point to the price python list
    for j in range(1,5):
        # v appending and casting to float data type
        Price.append(float(test[j]))
# v reversing the list to go from 9:30 am to 4:00 pm
Price.reverse()
# v printing the price python list
print(Price)
#for line 17, reason why i did from 1 to 5 was because the next data point was the price at 4:00 pm THURSDAY
# v creationg a file object to open the target file in write mode
textfile = open(f_name,"w")
# v writing the company name and the ticker id as the first line in the output text file
textfile.write(str(company) + " Ticker Id: " + str(TickerId) + "\n")
# v for loop to add data to file
for i in range(len(Price)):
    # v writing the data to the file, one price per line. converted to string in order to use \n escape key
    textfile.write(str(Price[i]) + "\n")
# v closing file stream
textfile.close()
#https://quotes-gw.webullfintech.com/api/search/pc/tickers?keyword=' + char + '&pageIndex=1&pageSize=1000
