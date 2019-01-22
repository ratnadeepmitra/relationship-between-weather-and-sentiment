# -*- coding: utf-8 -*-
### ANLY-501-02
### PROJECT 1
### GROUP 4
### Yipin, April, Kendra, Ratnadeep


#This code reads in the merged weather file (2017_weather_merged.csv) and
#prepares and cleans the data


## Import packages
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def cleanweather(df_weather):
    
    # Record dirty dataframe size
    n_row_dirty = df_weather.shape[0]
    n_col_dirty = df_weather.shape[1]
    

    ###---#0 Limit to US and surrounding stations only by lat and lon
    west_lon_limit=-163.843013
    east_lon_limit=-65.472298
    north_lat_limit=71.605384
    south_lat_limit=18.807327

    sel = ((df_weather['lon'] > west_lon_limit) & (df_weather['lon'] < east_lon_limit) & 
           (df_weather['lat'] > south_lat_limit) & (df_weather['lat'] < north_lat_limit))
    df_weather = df_weather[sel]

    ## Check new max and mins
    print('Latitude is now in the range: %f to %f' % (df_weather['lat'].min(), df_weather['lat'].max()))
    print('Longitude is now in the range: %f to %f' % (df_weather['lon'].min(), df_weather['lon'].max()))

    ## Map results to show density of remaining stations 
    ## Note: may run slow
    ## plt.scatter(df_weather['lon'], df_weather['lat'], s=1, alpha=0.1)
    ## plt.title('Plot of all NOAA Weather Stations')


    ###---#1 Clean datatypes
    ## Pre-cleaning: count number of unique datatype values and number of rows
    pre_u = df_weather['datatype'].unique().shape[0]
    pre_r = len(df_weather.index)

    ## Identify datatypes of interest and drop rows
    to_keep = ['PRCP','SNOW','TMAX','TMIN','TAVG']
    df_weather = df_weather[df_weather['datatype'].isin(to_keep)]
    df_weather = df_weather.reset_index(drop='True') #reset index

    ## Post-cleaning: count number of unique datatype values and number of rows
    post_u = df_weather['datatype'].unique().shape[0]
    post_r = len(df_weather.index)

    print('After cleaning datatypes, dropped', pre_r - post_r, 'rows and',
      pre_u - post_u, 'unique datatypes. Dataframe now has ', post_r, 'rows and ',
      post_u, 'unique values in the datatype column.')


    ###---#2 Drop rows with questionable quality
    ## Column 5 has a quality flag indicator, drop all those rows
    pre_r = len(df_weather.index)
    df_weather = df_weather[df_weather['c5'].isnull()]
    post_r = len(df_weather.index)

    print('After cleaning based on quality flag, dropped', pre_r - post_r, 
      'rows. Dataframe now has ', post_r, 'rows.')


    ##---#3 Drop unnecessary columns
    pre_c = df_weather.shape[1]
    to_keep = ['stationid', 'date', 'datatype', 'value', 'lat', 'lon']
    df_weather = df_weather[to_keep]
    post_c = df_weather.shape[1]

    print('Dropped', pre_c - post_c, 
         'unnecessary columns. Dataframe now has ', post_c, 'columns.')


    ##---#4 Clean stationids
    ## Reshape data
    df_weather = df_weather.pivot_table(index=['stationid','date','lat','lon'],columns='datatype',values='value')
    df_weather.reset_index(level=['date','stationid','lat','lon'], inplace=True)
    df_weather.columns.name = None

    ## Remove rows that do not have values for PRCP, TMAX and TMIN or TAVG
    df_weather = df_weather[df_weather['PRCP'].notnull()]
    df_weather = df_weather[(df_weather['TMAX'].notnull() & df_weather['TMIN'].notnull()) 
    | df_weather['TAVG'].notnull()]

    print('After reshaping, the dataframe now has %f rows and %f columns' % 
          (df_weather.shape[0], df_weather.shape[1]))


    ##---#5 Convert date format
    pre_type = df_weather['date'].dtype
    df_weather['date'] = pd.to_datetime(df_weather['date'].astype(str), format='%Y%m%d')
    post_type = df_weather['date'].dtype

    print('Converted date column type from: ', pre_type, 
         'to:', post_type)


    ##---#6 Convert temperature value formats
    ## Temp values given in tenths of degrees celcius so convert to celcius
    pre_max_tmin = df_weather['TMIN'].max()
    pre_max_tmax = df_weather['TMAX'].max()
    pre_max_tavg = df_weather['TAVG'].max()

    df_weather['TMIN'] = df_weather['TMIN'].apply(lambda x: x/10) 
    df_weather['TMAX'] = df_weather['TMAX'].apply(lambda x: x/10)
    df_weather['TAVG'] = df_weather['TAVG'].apply(lambda x: x/10)

    post_max_tmin = df_weather['TMIN'].max()
    post_max_tmax = df_weather['TMAX'].max()
    post_max_tavg = df_weather['TAVG'].max()

    print('Before converting, max of TMIN was: %f and after is: %f' % (pre_max_tmin, post_max_tmin))
    print('Before converting, max of TMAX was: %f and after is: %f', (pre_max_tmax, post_max_tmax))
    print('Before converting, max of TAVG was: %f and after is: %f', (pre_max_tavg, post_max_tavg))


    ##---#7 Data quality checks and more cleaning

    ## 7a# Check temperature values are valid
    ## Note: extreme values also checked by hand, all were accurate (and had accompanying news stories for breaking records!)
    print('TMIN is in the range: %f to %f' % (df_weather['TMIN'].min(), df_weather['TMIN'].max()))
    print('TMAX is in the range: %f to %f' % (df_weather['TMAX'].min(), df_weather['TMAX'].max()))
    print('TAVG is in the range: %f to %f' % (df_weather['TAVG'].min(), df_weather['TAVG'].max()))
    print('Expected temperature range in US is around (-15C, 35C)')

    ## 7b# Check TMIN is less than or equal to TMAX for when not missing
    ## Drop rows that do not meet criteria
    pre_r = len(df_weather.index)
    df_weather = df_weather[df_weather['TMIN'].notnull() <=  df_weather['TMAX'].notnull()]
    df_weather= df_weather.reset_index(drop='True') #reset index
    post_r = len(df_weather.index)

    print('Dropped', pre_r - post_r, 
      'rows with bad data. Dataframe now has ', post_r, 'rows.')

    ## 7c# Check PRCP values are valid
    ## Note: data is in tenths of mm
    ## Note: hand checked some extreme values, all were accurate (for ex. Hurricane Harvey)
    print('PRCP is in the range: %f to %f' % (df_weather['PRCP'].min(), df_weather['PRCP'].max()))
    print('Values appear accurate based on ncdc.noaa.gov information and checks by hand')

    ## 7d# Check all dates are vaild
    print('Dates are in the range of:', df_weather['date'].min(), 'to:',
        df_weather['date'].max())


    ##---#8 Check for outliers with histograms
    plt.hist(df_weather['TMIN'][df_weather['TMIN'].notnull()], bins=100)
    plt.title('Histogram of TMIN')

    plt.hist(df_weather['TMAX'][df_weather['TMAX'].notnull()], bins=100)
    plt.title('Histogram of TMAX')

    plt.hist(df_weather['TAVG'][df_weather['TAVG'].notnull()], bins=100)
    plt.title('Histogram of TAVG')

    plt.hist(df_weather['PRCP'], bins=100)
    plt.title('Histogram of PRCP')


    ##---#9 Missing value checks
    columns = df_weather.columns
    for c in columns:
        missing = df_weather[c].isnull().sum()
        print('In column:', c, 'there are:', missing, 'missing values.')

    ## Check that every row has at least (TMIN and TMAX) or (TAVG) by checking that they are never both null
    (df_weather['TMIN'].isnull() & df_weather['TAVG'].isnull()).value_counts()
    (df_weather['TMAX'].isnull() & df_weather['TAVG'].isnull()).value_counts()

    print('After checking, confirmed that every row has at least (TMIN and TMAX) or (TAVG)')


    ##---#10 Cleaned data summary
    nrow = df_weather.shape[0]
    ncol = df_weather.shape[1]
    print('After all cleaning, the dataframe has %d rows and %d columns.' % (nrow, ncol))

    df_weather.head()
    
    return df_weather

    
def main(args):
    ## Read in data as pandas dataframe, set column names
    filename_weather = '2017_Weather_RawData.csv'
    df_weather = pd.read_csv(filename_weather)
    df_weather = df_weather[['stationid', 'date', 'datatype', 'value',
                             'c4', 'c5', 'c6','c7', 'lat','lon']]
    
    ## Use df_wc to record cleaned weather dataframe
    start_time_clean_weather_data = datetime.now()
    df_wc = cleanweather(df_weather)

    ##---Save cleaned weather dataframe---##
    wc_outpath = '2017_Weather_CleanData.csv'
    with open(wc_outpath, 'w') as weather_cleaned:
        df_wc.to_csv(wc_outpath)
    weather_cleaned.close()
    
    end_time_clean_weather_data = datetime.now()
    print("Time taken to clean weather data: ", end_time_clean_weather_data - start_time_clean_weather_data)
    ##---Data Prep and Cleaning---##

if __name__=="__main__":
    main(sys.argv)
