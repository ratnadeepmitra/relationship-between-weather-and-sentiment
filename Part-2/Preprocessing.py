# -*- coding: utf-8 -*-
### ANLY-501-02
### PROJECT 2
### GROUP 4
### Yipin, April, Kendra, Ratnadeep

import sys
import os
import warnings
import pandas as pd
from geopy.distance import vincenty


def main(args):
    warnings.simplefilter('ignore', UserWarning)
    
    df = do_preprocessing()
    print(df)

def do_preprocessing():
    ###--read the cleaned data
    df_review = pd.read_csv('Restaurant_Reviews_CleanData.csv', encoding = 'Latin1')
    df_weather = pd.read_csv('2017_Weather_CleanData.csv', encoding = 'Latin1')
    df_detail = pd.read_csv('Restaurant_Details_CleanData.csv', encoding = 'Latin1')

    ###--- merge restaurant data (merge df_review to df_detail on id)
    #drop duplicates in df_detail that have same id
    df_detail = df_detail.drop_duplicates('id')

    #rename the columns in df_detail
    df_detail=df_detail.rename(columns={'coordinates.latitude':'lat','coordinates.longitude':'lon'})

    #compute merge
    df_rest = pd.merge(df_review, df_detail, left_on='restaurant_id', right_on='id', how='left')
    df_rest.shape


    ###--- match restaurants with station id by date

    #make dictionary of all dates in df_weather by splitting stationids by date (this takes ~30 secs)
    weather_by_date = {}
    for date in df_weather['date'].drop_duplicates():
        weather_by_date[date] = df_weather[df_weather['date'] == date].reset_index(drop=True)

    #create a dataframe to record the distance between resturant and station
    df_distance = pd.DataFrame(columns=['stationid','stationlat','stationlon','restid',
                                        'restlat','restlon','restname','restloc','distance','date'])
  
    #set counter for how many cannot find weather station match within +/-1
    counter = 0 

    #compute loop
    for j in (range(len(df_rest))):
    
        rest_date = df_rest.loc[j,'created_date']
        rest_lat  = df_rest.loc[j,'lat']
        rest_lon  = df_rest.loc[j,'lon']
    
        if rest_date in weather_by_date:
            todays_stations = weather_by_date[rest_date]
        
            sel = (
                    (todays_stations['lat'] > rest_lat - 1) &
                    (todays_stations['lat'] < rest_lat + 1) &
                    (todays_stations['lon'] > rest_lon - 1) &
                    (todays_stations['lon'] < rest_lon + 1)
                    )
        
            close_stations = todays_stations[sel]
    
            if len(close_stations) == 0:
                counter = counter + 1
                print('\n could not match ' + str(j))
            else:
                min_distance = 100000 #set a large number to start
                for i in range(len(close_stations)):
                    distance = vincenty((close_stations.iloc[i]['lat'], close_stations.iloc[i]['lon']),
                        (rest_lat, rest_lon)).miles
            
                    if distance < min_distance:
                        best_i = i
                        min_distance = distance
        
                df_distance = df_distance.append(pd.DataFrame([[
                    close_stations.iloc[best_i]['stationid'],
                    close_stations.iloc[best_i]['lat'],
                    close_stations.iloc[best_i]['lon'],
                    df_rest.iloc[j]['id'],
                    df_rest.iloc[j]['lat'],
                    df_rest.iloc[j]['lon'],
                    df_rest.iloc[j]['name'],
                    df_rest.iloc[j]['location'],
                    min_distance,
                    rest_date
                    ]],
                columns=['stationid','stationlat','stationlon','restid',
                            'restlat','restlon','restname','restloc','distance', 'date']))

    #end loop, show number of unmatched restaurants
    print(counter)

    #drop duplicates and reset index
    df_distance = df_distance.drop_duplicates().reset_index()
    del df_distance['index']

    #save to file
    df_distance.to_csv('distance.csv')

    #check results
    df_distance.head()
    df_distance['distance'].min() #min value
    df_distance['distance'].median() #median value
    df_distance['distance'].max() #max value
    df_distance['distance'].mean()
    #check the results
    print('The number of distances between restaurants and stations =',
    str(len(pd.unique(df_distance['restid']))))

    
    ###--- create merged dataframe from df_distance, df_rest, and df_weather
    df_distance.shape
    df_rest.shape

    len(df_rest['id'].unique())
    len(df_distance['restid'].unique())

    #compute merge
    df = pd.merge(df_rest, df_distance, left_on=['id', 'created_date'],
              right_on=['restid', 'date'])

    #compute merge to add in weather data
    df = pd.merge(df, df_weather, 
               left_on=['stationid', 'date'],
               right_on=['stationid', 'date']
              )

    #remove redundant columns
    del df['created_date']
    del df['id']
    #del df['Unnamed: 0']
    del df['restlat']
    del df['restlon']
    del df['name']
    del df['restid']
    del df['location']
    del df['lat_y']
    del df['lon_y']

    #rename columns
    df=df.rename(columns={ 'lat_x': 'rest_lat',
                         'lon_x': 'rest_lon',
                         'distance':'dist',
                         'location.city':'city',
                         'location.country':'country',
                         'location.state':'state',
                         'location.zip_code':'zip',
                         'restaurant_id':'restid',
                         'rating_x':'review_rating',
                         'rating_y':'rest_rating'})

    #check results and redudant columns
    df.head()
    df.shape
    df.columns
    
    #drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    #save to file
    df.to_csv('merged_data.csv')
    


if __name__ == "__main__":
    ## Execute only if run as a script  
    main(sys.argv)