
# -*- coding: utf-8 -*-
### ANLY-501-02
### PROJECT 1
### GROUP 4
### Yipin, April, Kendra, Ratnadeep

import pandas as pd
import numpy as np
import sys
from datetime import datetime

def main(args):
    
    start_time_clean_restaurant_data = datetime.now()
    clean_restaurant_data()
    end_time_clean_restaurant_data = datetime.now()
    print("Time taken to clean restaurant data: ", end_time_clean_restaurant_data - start_time_clean_restaurant_data)


def clean_restaurant_data():

    ## Import csv yelp data (Restaurant Details)
    print('Cleaning Restaurant Details')
    yelp_Details = pd.read_csv("Restaurant_Details_RawData.csv", encoding = 'latin1')

    list(yelp_Details)

    keepdetails = ['categories', 'coordinates.latitude', 'coordinates.longitude', 'id', 'location.address1',
                   'location.address2', 'location.address3', 'location.city', 'location.country', 'location.state',
                   'location.zip_code', 'name', 'price', 'rating', 'review_count', 'transactions']
    yelp_d = yelp_Details[keepdetails]
    
    ## Drop businesses that do not have any reviews
    yelp_d = yelp_d[yelp_d['review_count']!=0]
    yelp_d.reset_index(drop = True)
    yelp_d.shape
    yelp_d[['location.address2', 'location.address3']] = yelp_d[['location.address2', 'location.address3']].replace(np.NaN, "")
    yelp_d['location'] = yelp_d['location.address1'] + ' ' + yelp_d['location.address2'] + ' ' + yelp_d['location.address3']
    yelp_d = yelp_d.drop(['location.address1','location.address2', 'location.address3'], axis = 1)
    
    listcol = list(yelp_d)
    for li in listcol:
        if yelp_d[li].isnull().values.any() == True:
            print(str(li) + " yes missing values")
        else:
            print(str(li) + " no missing values")

    ## Identify missing longitude and latitude values   
    print(yelp_d[yelp_d['coordinates.latitude'].isnull()])
    print("The restaurant name " + str(yelp_d[yelp_d['coordinates.latitude'].isnull()]['name'].values[0]) + " is missing long and lat values")

    ## Find longitude and latitude values using restaurant name and address through google maps
    yelp_d['coordinates.latitude'][yelp_d['name']=='Milya cafe'] = 39.266846
    yelp_d['coordinates.longitude'][yelp_d['name']=='Milya cafe'] = -84.379769

    ## Clean transactions variable by stripping off the brackets       
    yelp_d['transactions'] = yelp_d['transactions'].str.strip('[]')

    ## Replace all the links under the transaction column with empty string
    changeTran = list(yelp_d[yelp_d['transactions'].str.startswith("https:")]['transactions'].index)
    for i in changeTran:
        print(yelp_d.iloc[i, 12])
        yelp_d.iloc[i, 12] = ''
        
    ## Write cleaned data to a .csv file
    yelp_d.to_csv('Restaurant_Details_CleanData.csv', index = False, mode = 'w', header = True)
        
    ## Import data (Restaurant Reviews)
    print('Cleaning Restaurant Reviews')
    revdf = pd.read_csv("Restaurant_Reviews_RawData.csv", encoding='latin-1')

    ## Drop columns that are not needed 
    print(list(revdf)) # Check list of columns before dropping
    revdf = revdf.drop(['url','user.image_url','user.name'], 1)
    print(list(revdf)) # Check list of columns after dropping
    
    ## Check for missing values in the columns
    columns = revdf.columns
    for c in columns:
        missing = revdf[c].isnull().sum()
        print('In column:', c, 'there are:', missing, 'missing values.')
   
    ## Split column based on white-space and removing the hour part
    ## Keep on the date created
    print('Type of data in date column initially: ',type(revdf['time_created'][1]))
    revdf["created_date"], revdf["created_time"] = revdf["time_created"].str.split(' ', 1).str
    revdf = revdf.drop(['created_time','time_created'], 1)
    revdf['created_date'] = revdf['created_date'].astype('datetime64[ns]')
    print('Type of data in date column after updating: ',type(revdf['created_date'][1]))
    
    ## Check for 2017 data and delete the rest
    revdf = revdf[revdf['created_date'] >= date(2017, 1, 1)]
    
    print(len(revdf))
    revdf = revdf.drop_duplicates(subset=['restid', 'text'], take_last=True)
    print(len(revdf))
    
    ## Write cleaned data to a .csv file
    revdf.to_csv('Restaurant_Reviews_CleanData.csv', index = False, mode = 'w', header = True)
    
    
if __name__ == "__main__":
	## Execute only if run as a script  
	main(sys.argv)


