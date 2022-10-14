import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from google.colab import files
files.upload()
def split_vegetation(data_year, brush, grass, timber, index_year, index_vegetation, index_acres, print_true):
    for i in range(0, len(data_year)):
          if data_year[i][index_vegetation].find("B") > -1:
                brush[index_year][0] = brush[index_year][0]+1
                if type(data_year[i][index_acres]) == float:
                    brush[index_year][1] = brush[index_year][1] + (float(data_year[i][index_acres])/len(data_year[i][index_vegetation]))
                else:
                    if data_year[i][index_acres].find(",") > -1:
                        brush[index_year][1] = brush[index_year][1] + (float(data_year[i][index_acres].replace(',', ''))/len(data_year[i][index_vegetation]))
                    else:
                        brush[index_year][1] = brush[index_year][1] + (float(data_year[i][index_acres])/len(data_year[i][index_vegetation]))
            if data_year[i][index_vegetation].find("G") > -1:
                grass[index_year][0] = grass[index_year][0]+1
                if type(data_year[i][index_acres]) == float:
                    grass[index_year][1] = grass[index_year][1] + (float(data_year[i][index_acres])/len(data_year[i][index_vegetation]))
                else:
                    if data_year[i][index_acres].find(",") > -1:
                        grass[index_year][1] = grass[index_year][1] + (float(data_year[i][index_acres].replace(',', ''))/len(data_year[i][index_vegetation]))
                    else:
                        grass[index_year][1] = grass[index_year][1] + (float(data_year[i][index_acres])/len(data_year[i][index_vegetation]))
            if data_year[i][index_vegetation].find("T") > -1:
                timber[index_year][0] = timber[index_year][0]+1
                if type(data_year[i][index_acres]) == float:
                    timber[index_year][1] = timber[index_year][1] + (float(data_year[i][index_acres])/len(data_year[i][index_vegetation]))
                else:
                    if data_year[i][index_acres].find(",") > -1:
                        timber[index_year][1] = timber[index_year][1] + (float(data_year[i][index_acres].replace(',', ''))/len(data_year[i][index_vegetation]))
                    else:
                        timber[index_year][1] = timber[index_year][1] + (float(data_year[i][index_acres])/len(data_year[i][index_vegetation]))
def fix_nan(data):
    for i in range(0, len(data[0])):
        for x in range(0, len(data)):
            if type(data[x][i]) == float and math.isnan(data[x][i]):
                data[x][i] = "None"
            elif type(data[x][i]) == None:
                data[x][i] = "None"
    return data

def find_unique(data, unique_values, index_column):
    count = 0
    for i in range(0, len(data)):
        unique = True
        if data[i][index_column].find("/") > -1 or data[i][index_column].find("-") > -1 or data[i][index_column].find(",") > -1:
            #count = count+1
            unique = False
        for x in range(0, len(unique_values)):
            if data[i][index_column].lower().replace(' ', '') == unique_values[x].lower().replace(' ', '') or data[i][index_column] == "Other Agencies":
                unique = False
        if unique:
            unique_values.append(data[i][index_column].upper())
    #return count

def values_by_county(county_array, year_data, index_year, index_acres):
    for i in range(0, len(county_array)):
        for x in range(0, len(year_data)):
            #print(x)
            #print(type(county_array[i][0]))
            if county_array[i][0].upper() == year_data[x][0].upper():
                if type(year_data[x][index_acres]) == float:
                    county_array[i][index_year+1] = county_array[i][index_year+1] + int(year_data[x][index_acres])
                else:
                    if year_data[x][index_acres].find(",") > -1:
                        county_array[i][index_year+1] = county_array[i][index_year+1] + int(year_data[x][index_acres].replace(',', ''))
                    else:
                        county_array[i][index_year+1] = county_array[i][index_year+1] + int(year_data[x][index_acres])

def find_cumulative(data_original):

    data_cumulative = np.zeros((len(data_original), len(data_original[0])), dtype='int')
    data_cumulative[0] = data_original[0]
    for i in range(1, len(data_original)):
        data_cumulative[i] = data_cumulative[i-1] + data_original[i]
    return data_cumulative
data_08 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2008-Fixed.csv")))
data_09 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2009-Fixed.csv")))
data_10 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2010.csv")))
data_11 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2011.csv")))
data_12 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2012.csv")))
data_13 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2013.csv")))
data_14 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2014.csv"), dtype="object"))
data_15 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2015.csv")))
data_16 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2016.csv")))
data_17 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2017.csv")))
data_18 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2018.csv")))
data_19 = fix_nan(np.array(pd.read_csv("Wildfire-Data-2019.csv")))
data_temp_08 = np.array(pd.read_csv("Average_Temperature_Data-2008.csv"))
print(data_08[:, :])
print(data_09[:, :])
brush = np.zeros((12, 2), dtype = 'float64')
grass = np.zeros((12, 2), dtype = 'float64')
timber = np.zeros((12, 2), dtype = 'float64')
split_vegetation(data_year=data_08, brush=brush, grass=grass, timber=timber, index_year=0, index_vegetation=3, index_acres=2, print_true=True)
split_vegetation(data_year=data_09, brush=brush, grass=grass, timber=timber, index_year=1, index_vegetation=3, index_acres=2, print_true=True)
split_vegetation(data_year=data_10, brush=brush, grass=grass, timber=timber, index_year=2, index_vegetation=2, index_acres=1, print_true=True)
split_vegetation(data_year=data_11, brush=brush, grass=grass, timber=timber, index_year=3, index_vegetation=2, index_acres=1, print_true=True)
split_vegetation(data_year=data_12, brush=brush, grass=grass, timber=timber, index_year=4, index_vegetation=2, index_acres=1, print_true=True)
split_vegetation(data_year=data_13, brush=brush, grass=grass, timber=timber, index_year=5, index_vegetation=3, index_acres=2, print_true=True)
split_vegetation(data_year=data_14, brush=brush, grass=grass, timber=timber, index_year=6, index_vegetation=3, index_acres=2, print_true=True)
split_vegetation(data_year=data_15, brush=brush, grass=grass, timber=timber, index_year=7, index_vegetation=3, index_acres=2, print_true=True)
split_vegetation(data_year=data_16, brush=brush, grass=grass, timber=timber, index_year=8, index_vegetation=3, index_acres=2, print_true=True)
split_vegetation(data_year=data_17, brush=brush, grass=grass, timber=timber, index_year=9, index_vegetation=3, index_acres=2, print_true=True)
split_vegetation(data_year=data_18, brush=brush, grass=grass, timber=timber, index_year=10, index_vegetation=3, index_acres=2, print_true=True)
split_vegetation(data_year=data_19, brush=brush, grass=grass, timber=timber, index_year=11, index_vegetation=3, index_acres=2, print_true=True)
brush_cumulative = find_cumulative(brush)
timber_cumulative = find_cumulative(timber)
grass_cumulative = find_cumulative(grass)
print(brush)
print("Time vs Brush Acres Total")
plt.plot(np.arange(2008, 2020), brush_cumulative[:, 1], 'o')
plt.show()
print("Time vs Brush Fires_total")
plt.plot(np.arange(2008, 2020), brush_cumulative[:, 0], 'o')
plt.show()
print("Time vs Timber Acres Total")
plt.plot(np.arange(2008, 2020), timber_cumulative[:, 1], 'o')
plt.show()
plt.plot(np.arange(2008, 2020), timber_cumulative[:, 0], 'o')
plt.show()
print("Grass")
plt.plot(np.arange(2008, 2020), grass_cumulative[:, 1], 'o')
plt.show()
plt.plot(np.arange(2008, 2020), grass_cumulative[:, 0], 'o')
plt.show()
print(brush_cumulative)
print(timber_cumulative)
print(grass_cumulative)
unique_values = []
find_unique(data_08, unique_values, 0)
find_unique(data_09, unique_values, 0)
find_unique(data_10, unique_values, 0)
find_unique(data_11, unique_values, 0)
find_unique(data_12, unique_values, 0)
find_unique(data_13, unique_values, 0)
find_unique(data_14, unique_values, 0)
find_unique(data_15, unique_values, 0)
find_unique(data_16, unique_values, 0)
find_unique(data_17, unique_values, 0)
find_unique(data_18, unique_values, 0)
find_unique(data_19, unique_values, 0)
print(unique_values)
fire_acres_by_county = np.array(unique_values, dtype='object')
fire_acres_by_county = fire_acres_by_county.reshape((len(fire_acres_by_county), 1))
for i in range(0, 12):
    fire_acres_by_county = np.append(fire_acres_by_county, np.zeros((len(unique_values), 1)), axis=1)
values_by_county(fire_acres_by_county, data_08, 0, 1)
values_by_county(fire_acres_by_county, data_09, 1, 1)
values_by_county(fire_acres_by_county, data_10, 2, 1)
values_by_county(fire_acres_by_county, data_11, 3, 1)
values_by_county(fire_acres_by_county, data_12, 4, 1)
values_by_county(fire_acres_by_county, data_13, 5, 2)
values_by_county(fire_acres_by_county, data_14, 6, 2)
values_by_county(fire_acres_by_county, data_15, 7, 2)
values_by_county(fire_acres_by_county, data_16, 8, 2)
values_by_county(fire_acres_by_county, data_17, 9, 2)
values_by_county(fire_acres_by_county, data_18, 10, 2)
values_by_county(fire_acres_by_county, data_19, 11, 2)
fire_acres_by_county_cumulative = np.zeros((len(fire_acres_by_county), len(fire_acres_by_county[0])), dtype='object')
for i in range(0, len(fire_acres_by_county)):
    fire_acres_by_county_cumulative[i][0] = fire_acres_by_county[i][0]
    fire_acres_by_county_cumulative[i][1] = fire_acres_by_county[i][1]
    for x in range(2, len(fire_acres_by_county[0])):
        fire_acres_by_county_cumulative[i][x] = fire_acres_by_county[i][x]+fire_acres_by_county_cumulative[i][x-1]
    for i in range(0, len(fire_acres_by_county_cumulative)):
        print(fire_acres_by_county[i][0])
        plt.plot(np.arange(2008, 2020), fire_acres_by_county_cumulative[i][1:13], 'o')
        plt.show() 
