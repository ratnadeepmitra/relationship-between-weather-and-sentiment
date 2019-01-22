# -*- coding: utf-8 -*-
### ANLY-501-02
### PROJECT 1
### GROUP 4
### Yipin, April, Kendra, Ratnadeep

import pandas as pd
import io
import requests
import sys
from datetime import datetime

def main(args):
    
    start_time_get_counties = datetime.now()
    county_df = get_counties_usa()
    county_df.to_csv('USA_Counties_RawData.csv', index = False, mode = 'w', header = True)
    end_time_get_counties = datetime.now()
    print("Time taken to collect counties data: ", end_time_get_counties - start_time_get_counties)
    
    start_time_clean_counties = datetime.now()
    county_df_clean = clean_counties_usa(county_df)
    county_df_clean.to_csv('USA_Counties_CleanData.csv', index = False, mode = 'w', header = True)
    end_time_clean_counties = datetime.now()
    print("Time taken to clean counties data: ", end_time_clean_counties - start_time_clean_counties)

def get_counties_usa():
    
    url = "https://www2.census.gov/programs-surveys/popest/datasets/2010-2016/counties/totals/co-est2016-alldata.csv"
    resp = requests.get(url).content
    county_df = pd.read_csv(io.StringIO(resp.decode('latin-1'))) 
    ## Sort the data in descending order
    county_df = county_df.sort_values('POPESTIMATE2016', ascending = False)
    ## Choose 300 counties based on population  
    county_df_2 = county_df.head(300)
    return(county_df_2)


def clean_counties_usa(county_df):
    
    print("Looking at the first 5 rows of the data")
    print(county_df.head())

    ## List of all the column names - don't need all of them
    ## Drop all the other columns
    listnames = list(county_df)
    print(listnames)
    keepNames = ['COUNTY', 'STNAME', 'CTYNAME', 'CENSUS2010POP', 'POPESTIMATE2015', 'POPESTIMATE2016']
    df_short = county_df[keepNames]
    
    ## Dimension of the dataframe
    print("Reduced the number of columns from 116 to 5")
    print(df_short.shape)
    
    ## Remove all the state aggregate data
    df_short = df_short[df_short['COUNTY'] != 0]
    
    ## Reset index
    df_short.reset_index(drop='True')
    
    print(df_short.describe())
        
    for nm in keepNames:
        if min(df_short[nm]) == 'NaN':
            text = str(nm) + "has missing data"
            print(text)
        else:
            text = str(nm) + " column looks good"
            print(text)
    
    ## Trimmed the word 'county' in the county name
    index0 = df_short.CTYNAME.str.split('\s+').str[0]
    
    ## Has county name
    index1 = df_short.CTYNAME.str.split('\s+').str[1]
    new1 = index1.replace("County", "")
    df_short['CountyName'] = index0 + ' ' + new1
    
    ## Check min and max numbers of population number
    checkmin = min(df_short.POPESTIMATE2016)
    text = "The smallest population estimate for 2016 is " + str(checkmin)
    print(text)    
    checkmax = max(df_short.POPESTIMATE2016)
    text = "The largest population estimate for 2016 is " + str(checkmax)
    print(text)
    
    ## Find the difference in population between 2015-2016
    difinPop = df_short.POPESTIMATE2016 - df_short.POPESTIMATE2015
    maxIndex = difinPop[difinPop == max(difinPop)].index.values[0]
    maxValue = difinPop[difinPop == max(difinPop)].values[0]
    
    ## Get the index of maximum and minimum between 2015 and 2016 population estimator
    ## Check to see if there was a huge jump in the population
    print("County " + str(df_short[df_short.index == maxIndex].CountyName.values[0]) + " has the biggest difference of " + str(maxValue))
    
    ## Percentage of increase
    maxPer = "{:.0%}".format(maxValue/df_short[df_short.index == maxIndex].POPESTIMATE2015.values[0])
    textMaxper = "The percentage increase of population for county " + str(df_short[df_short.index == maxIndex].CountyName.values[0]) +  " is ~" + str(maxPer)
    print(textMaxper)
    
    ## Sort the data in descending 
    countyList = df_short.sort_values('POPESTIMATE2016', ascending = False)
    
    ## Selecting the two relevant columns and renaming them
    countyList_2 = countyList[['CTYNAME','STNAME']]
    countyList_2.columns = ['county','state']
    ## Final list of county names
    return(countyList_2[['county','state']].head(300))
    

if __name__ == "__main__":
	## Execute only if run as a script  
	main(sys.argv)





