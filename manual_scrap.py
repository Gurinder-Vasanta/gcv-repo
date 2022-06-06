import requests
link = str(input("Enter request URL: "))
t_file = str(input('Enter target file: '))
temp = requests.get(link).json()
textfile = open(t_file, "w")
textfile.write('Prices\n')
for i in range(len(temp)):
    textfile.write(str(temp[i]["Close"]) + "\n")
