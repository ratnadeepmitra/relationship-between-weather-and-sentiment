# -*- coding: utf-8 -*-
### ANLY-501-02
### PROJECT 1
### GROUP 4
### Yipin, April, Kendra, Ratnadeep

import sys
import pandas as pd
import requests  
import warnings
import json
import ftplib
from datetime import datetime
from pandas.io.json import json_normalize

def main(args):
    warnings.simplefilter('ignore', UserWarning)
    
    ## Get the weather data
    ## Data gets stored in .csv file
    get_weather_data()
    
    ## Create the output file beforehand
    ## Henceforth data can be appended to it
    file1 = open('Restaurant_Details_RawData.csv','w') 
    file1.close() 
    file2 = open('Restaurant_Reviews_RawData.csv','w') 
    file2.close()

    ## Import list of counties
    county_df = pd.read_csv('USA_Counties_CleanData.csv', encoding = 'Latin1')
    county_df['county_state'] = county_df['county'] + ' ' + county_df['state']
    county_state_names = county_df['county_state']
    county_df.to_csv('USA_Counties_RawData.csv', index = False, mode = 'w', header = True)
    
    ## Obtain these from Yelp's manage access page
    app_id = 'g3jia6zziSd4sAOHry3OPQ'
    app_secret = 'soqkh03A7KBco6hKnKCxbFkbggTd9UXnGdTNFXuDcB6Yt74ajx1QFq8guEN8WBKW'
    data = {'grant_type': 'client_credentials',
            'client_id': app_id,
            'client_secret': app_secret}
    token = requests.post('https://api.yelp.com/oauth2/token', data = data)
    access_token = token.json()['access_token']
    count_county = 1
    review_count = 1
    
    start_time_get_yelp_data = datetime.now()
    for county_name in county_state_names:
        response_data_restaurant_details = get_restaurant_details(access_token, county_name) 
           
        ## Get the reviews of each of the restaurants
        if 'businesses' in response_data_restaurant_details:
            restaurant_df = json_normalize(response_data_restaurant_details['businesses'])
            if count_county == 1:
                restaurant_df.to_csv('Restaurant_Details_RawData.csv', index = False, mode = 'a', header = True)
            else:
                restaurant_df.to_csv('Restaurant_Details_RawData.csv', index = False, mode = 'a', header = False)
            for item in response_data_restaurant_details['businesses']:
                response_data_restaurant_reviews = get_restaurant_reviews(access_token, item['id'])
                if 'reviews' in response_data_restaurant_reviews:
                    restaurant_reviews = response_data_restaurant_reviews['reviews']
                    for rvw in restaurant_reviews:
                        rvw['restaurant_id'] = item['id']
                        review_df = json_normalize(rvw)   
                        if count_county == 1 & review_count == 1:
                            review_df.to_csv('Restaurant_Reviews_RawData.csv', index = False, header = True, mode = 'a')
                            review_count = review_count + 1
                        else:
                            review_df.to_csv('Restaurant_Reviews_RawData.csv', index = False, header = False, mode = 'a')                        
        count_county = count_county + 1
    end_time_get_yelp_data = datetime.now()    
    print("Time taken to collect restaurant data from Yelp: ", end_time_get_yelp_data - start_time_get_yelp_data)
    
def get_restaurant_details(access_token, county_name):  
    ## Base URL
    url = 'https://api.yelp.com/v3/businesses/search'
    headers = {'Authorization': 'bearer %s' % access_token}
    params = {'location': county_name,
          'term': 'restaurants',
          'sort': '0'
          }
    try:
        resp = requests.get(url = url, params = params, headers = headers).content
        ## Transform the JSON API response into a Python dictionary
        data = json.loads(resp.decode("utf-8"))
        return data
    except json.decoder.JSONDecodeError:
        return ''

def get_restaurant_reviews(access_token, res_id):
    ## Base URL
    url = 'https://api.yelp.com/v3/businesses/' + res_id + '/reviews'
    headers = {'Authorization': 'bearer %s' % access_token}
    try:
        resp = requests.get(url = url, headers = headers).content
        ## Transform the JSON API response into a Python dictionary
        data = json.loads(resp.decode("utf-8"))
        return data
    except json.decoder.JSONDecodeError:
        return ''


def get_weather_data():
    
    start_time_get_weather_data = datetime.now()
    
    ## Use FTP to get 2017 file from NOAA's GHCND dataset
    ## Login to ftp server
    ftp = ftplib.FTP('ftp.ncdc.noaa.gov', 'anonymous')

    ## Go to daily by year file
    ftp.cwd('/pub/data/ghcn/daily/by_year/')

    ## Grab file
    filename = '2017.csv.gz'
    with open(filename, 'wb') as local_file:
        ## Write data to file
        ftp.retrbinary('RETR ' + filename, local_file.write)
    local_file.close()

    ## Read in data as pandas dataframe
    df_weather = pd.read_csv(filename, header = None)
    
    ## Go to daily directory and grab location reference file, write to local file
    ftp.cwd('/pub/data/ghcn/daily/')
    filename2 = 'ghcnd-stations.txt'
    with open(filename2, 'wb') as local_file2:
        ## Write data to file
        ftp.retrbinary('RETR ' + filename2, local_file2.write)
    local_file2.close()
    
    ftp.quit()
    
    ## Read in location reference file as pandas dataframe from flat-width-file
    df_wlf = pd.read_fwf(filename2, widths = (11, 10, 10, 1000), header = None)
    df_wlf.columns = ('stationid', 'lat', 'lon', 'everythingelse')
    df_wlf = df_wlf[['stationid', 'lat', 'lon']]
    df_wlf.shape
        
    ## Rename the columns
    df_weather.columns = ('stationid', 'date', 'datatype', 'value', 'c4', 'c5', 'c6','c7')
    
    ## Merge dataframes such that lat and long are addeded for each station location
    df_weather = pd.merge(left = df_weather, right = df_wlf, on = 'stationid')
    df_weather.shape
    df_weather.head()

    ## Save new merged dataframe as file
    outpath = '2017_Weather_RawData.csv'
    with open(outpath, 'w') as weather_merge:
        df_weather.to_csv(outpath)
    weather_merge.close()
    
    end_time_get_weather_data = datetime.now()
    print("Time taken to get weather data: ", end_time_get_weather_data - start_time_get_weather_data)
    
if __name__ == "__main__":
	## Execute only if run as a script  
	main(sys.argv)
