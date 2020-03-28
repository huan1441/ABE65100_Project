# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Group Name: Hydrologic Modeling
#
# Created by Mingda Lu, Tianle Xu, Linji Wang, Tao Huang
#            March 26, 2020, in Python 3.6
#
# Script to conduct a graphical analysis and data quality checking of
# hydrologic data (streamflow and precipitation)
#      
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Import the required Modules/Packages for obtaining the data from portal
import sys, os
import urllib.parse
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from cdo_api_py import Client
import pandas as pd
from datetime import datetime
import scipy as si
from scipy import stats
import pylab as py 

# 1.# Define a function for obtaining the daily flow data from USGS Surface Data Portal
## Parameters - Station number and folder name
def GetPeakFlowYear(station_number, FolderName, begin_date, end_date):
    ## Building URLs
    var1 = {'site_no': station_number}
    var2 = {'begin_date': begin_date}
    var3 = {'end_date': end_date}
    part1 = 'https://nwis.waterdata.usgs.gov/nwis/dv?'
    part2 = 'cb_00060=on&format=rdb&'
    part3 = '&referred_module=sw&period=&'
    part4 = '&'
    link = (part1 + part2 + urllib.parse.urlencode(var1) + part3 + urllib.parse.urlencode(var2) + part4 + urllib.parse.urlencode(var3))
    print('The USGS Link is: \n',link)
    
    ## Opening the link & retrieving data
    response = urllib.request.urlopen(link)
    page_data = response.read()
    
    ## File name assigning & storing the rav data as text file
    with open(FolderName+'Data_' + station_number + '_raw' + '.csv', 'wb') as f1:
        f1.write(page_data)
    f1.close

# define and initialize the missing data dictionary
ReplacedValuesDF = pd.DataFrame(0, index=["1. No Data","2. Gross Error"],
                                columns=['Discharge','Precipitation'])

## Main Code
station_number=input('Enter UHC8 Number of the required Station (USGS Station Number/site_no) \t') #04180000
begin_date=input('Enter begin date (format:yyyy-mm-dd) \t') #2019-01-01
end_date=input('Enter end date (format:yyyy-mm-dd) \t') #2019-12-31
print('\t')

## Assigning the location for storing the data
## First Method
FolderName='./Results/'
if os.path.exists(FolderName) == False:
    os.mkdir(FolderName)

dailyflow_list_wb=GetPeakFlowYear(station_number,FolderName,begin_date,end_date)

# read discharge data and give name for needed columns 
data = pd.read_csv(FolderName+'Data_' + station_number + '_raw' + '.csv',skiprows=30,
header=None,sep='\t',usecols=[2,3],names=['Timestamp','Daily_Discharge'])

data = data.set_index('Timestamp')

# Check01 Remove No Data values
# record the number of values with NaN and delete them

ReplacedValuesDF['Discharge'][0]= data.isna().sum()

data = data.dropna()

# Check02 Check for Gross Errors
# find the index of the gross errors for Discharge and record the number
# the threshold is Q ≥ 0

index=(data>=0)

ReplacedValuesDF['Discharge'][1]= len(index)-index.sum()

# delete the gross errors 

data = data[index].dropna()

## Graphical analysis after data quality checking
# plot daily streamflow hydrograph
data.plot(figsize=(15,7))
plt.xlabel('Time')
plt.ylabel('Discharge (cfs)')
plt.title('Daily Discharge Hydrograph')
plt.legend(["Discharge"])
plt.savefig(FolderName + "Daily Discharge.jpeg")
plt.close()

## QQplot 
plt.figure(figsize=(15,7))
data_points=data.Daily_Discharge
si.stats.probplot(np.log(data_points), dist='norm', plot=plt)
plt.savefig(FolderName + "Q-Q Plot of Discharge.jpeg")
plt.close()

## Boxplot
plt.figure(figsize=(15,7))
plt.boxplot(data.Daily_Discharge,whis=3)
plt.savefig(FolderName + "Boxplot of Discharge.jpeg")
plt.close()

print("Discharge data processing is done!")


# 2. # Extract rainfall data
token = input('Enter into the token to access to the data: \t') #'lsANjWwoJQegJhKZtKNJPVDGWIGhBSJN'
# the Client object helps you acess the NCDC database with your token
my_client = Client(token, default_units='None', default_limit=1000)

# The extend is the lat, long of the target region.
extent = dict()
Dirs = ['north','south','east','west']
NSEW = input('Enter the extent, format:"N,S,E,W":')
temp = NSEW.split(',')
for i in range(len(Dirs)):
    extent[Dirs[i]] = float(temp[i])
    
# Displaying the dictionary
for key, value in extent.items():
	print(str(key)+':'+str(value)) #extent = 41.53,41.21,-84.90,-85.33
    
# input of start data, end date, type of dataset, and name of gauge
start_date = input('Enter begin date (format:yyyy-mm-dd) \t') # 2019-01-01
startdate = pd.to_datetime(start_date)
end_date = input('Enter end date (format:yyyy-mm-dd) \t') #2019-12-31
enddate = pd.to_datetime(end_date)

datasetid = 'GHCND'

#The find_station function returns the dataframe containing stations' info within the input extent.
stations = my_client.find_stations(
           datasetid = datasetid,
           extent = extent,
           startdate = startdate,
           enddate = enddate,
           return_dataframe = True)

# download data from all stations specified by their id
for i in stations.id:
    Rainfall_data = my_client.get_data_by_station(datasetid = datasetid, stationid = i,
               startdate = startdate, enddate = enddate, return_dataframe = True,
               include_station_meta = True)
    station_id = i.split(":")[1]
    filename = datasetid + '_' + station_id + '.csv'
    Rainfall_data.to_csv(FolderName+filename)
    
# Get the daily average values of  all stations
P=[]
for i in stations.id:
    station_id = i.split(":")[1]
    filename = datasetid + '_' + station_id + '.csv'
    df = pd.read_csv(FolderName+filename)
    df = df[['date','PRCP']]
    P.append(df)
data=pd.concat(P,axis=0,ignore_index=True)
data.to_csv(FolderName+'P.csv')
data = data.set_index('date')

# Check01 Remove No Data values
# record the number of values with NaN and replace with zero

ReplacedValuesDF['Precipitation'][0]= data.isna().sum()

data = data.fillna(0)

# Check02 Check for Gross Errors
# find the index of the gross errors for Discharge and record the number
# the threshold is 0≤ P ≤ 1870mm (the max daily rainfall in the world)

# unit of raw data is 0.1mm
index=(data>=0) & (data<=18700)

ReplacedValuesDF['Precipitation'][1]= len(index)-index.sum()

# delete the gross errors 

data = data[index].dropna()

data=data.groupby('date')['PRCP'].mean()

# cover the unit into inch
data=data/10/25.4
data.index = pd.to_datetime(data.index)

# plot monthly pricipitation hydrograph
monthly_data=data.resample('M').sum()
monthly_data.index = monthly_data.index.strftime('%Y-%m')
monthly_data.plot.bar(figsize=(15,7))
plt.ylabel('Precipitation (in)')
plt.xlabel('Time')
plt.title('Hyetograph')
plt.savefig(FolderName + "Hyetograph.jpeg")
plt.close()

print("Precipitation data processing is done!")


## Plot the return period by peak flow data
## Define a function for obtaining the peak flow data from USGS Surface Data Portal
## Parameters - Station number and folder name
def GetPeakFlowYear(station_number, FolderName):
    ## Building URLs
    var1 = {'site_no': station_number}
    part1 = 'https://nwis.waterdata.usgs.gov/nwis/peak?'
    part2 = '&agency_cd=USGS&format=rdb'
    link = (part1 + urllib.parse.urlencode(var1) + part2)
    print('The USGS Link is: \n',link)
    
    ## Opening the link & retrieving data
    response = urllib.request.urlopen(link)
    page_data = response.read()
    
    ## File name assigning & storing the rav data as text file
    with open(FolderName+'Data_' + station_number + '_raw' + '.csv', 'wb') as f1:
        f1.write(page_data)
    f1.close
    
## Main Code
station_number=input('Enter UHC8 Number of the required Station (USGS Station Number/site_no) \t')
print('\t')
## Assigning the location for storing the data
## First Method
FolderName='./Results/'
## Second Method
#Foldername='/home/mygeohub/xu1361/DA1/Results/'
peakflow_list_wb=GetPeakFlowYear(station_number,FolderName)

# read discharge data and give name for needed columns 
data = pd.read_csv(FolderName+'Data_' + station_number + '_raw' + '.csv',skiprows=74,
header=None,sep='\t',usecols=[2,4],na_filter=True,names=['Timestamp','Peak_Discharge'])

# delete empty data
data = data.dropna()
data = data.reset_index(drop=True)

# get the values of mean and standard derivation of peak discharge data 
discharge_mean=np.mean(data['Peak_Discharge'])
discharge_sd=np.std(data['Peak_Discharge'])

# calculate the streamflow of different return years 
ReturnPeriod = [10, 25, 50, 100, 500]
StreamFlow = []
for i in ReturnPeriod:
    a = discharge_mean - (math.sqrt(6) / math.pi) * (0.5772 + math.log (math.log ( i / (i-1)))) * discharge_sd
    StreamFlow.append(a)
    
# plot streamflow hydrograph
plt.figure(figsize=(15,7))
plt.plot(ReturnPeriod,StreamFlow)
plt.xlabel('Return Period (yrs)')
plt.ylabel('Discharge (cfs)')
plt.title('Extreme Discharge Hydrograph')
plt.savefig(FolderName + "Extreme Discharge Hydrograph.jpeg")
plt.close()

print("Return period discharge data processing is done!")


## create a new output file to store the checked results

ReplacedValuesDF.to_csv("Checked_Summary.txt", sep="\t")

print("All data processing is done!")
