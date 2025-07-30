
import pandas as pd
import numpy as np
import os
import re
import time

#Load from 
from project_setup import PROJECT_DATA_DIR, DATA_DIR, TS_DIR

# Function to load avvail data 
def load_avvail_data(data_dir):
    """
    Load all the json files into a single dataframe
    """
    
    files = [os.path.join(data_dir,file) for file in os.listdir(data_dir) if re.search('\.json',file)]
    
    df_list = [pd.read_json(file) for file in files]
    df_list = [df.rename(columns = {'StreamID' : 'stream_id',
                                    'TimesViewed' : 'times_viewed',
                                    'total_price': 'price'}) for df in df_list]
    
    df = pd.concat(df_list, sort = True)
    
    
    df['dates'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['year_month'] = [str(date)[0:7] for date in df.dates]

    return df



def timeseries_aggregate(dataframe, country = None):
    """
    Aggregate data into a daily timeseries. Country can be selected to generate data
    for only a single country
    """

    if country != None:
        dataframe = dataframe[dataframe.country == country]


    #Create a date range to ensure no dates are missing from the data.
    first_date = dataframe['dates'].min()
    last_date = dataframe['dates'].max()
    date_range  = pd.date_range(first_date, last_date, freq = 'D')
    
    
    daily_revenue = [dataframe[dataframe['dates'] == days]['price'].sum() for days in date_range]
    daily_views = [dataframe[dataframe['dates'] == days]['times_viewed'].sum() for days in date_range]
    daily_purchases = [dataframe[dataframe['dates'] == days]['price'].size for days in date_range]
    
    
    dataframe_dict= {
        'daily_revenue' : daily_revenue,
        'daily_views' : daily_views,
        'daily_purchases': daily_purchases,
        'dates': date_range
    }    
    
    timeseries_df = pd.DataFrame(dataframe_dict)
    
    return timeseries_df


def load_ts(ts_dir = TS_DIR, data_dir = DATA_DIR, replace = False, verbose = True):
    """
    Function to read in timeseries formatted data. Option to replace/recreate files if the user desires. Set 
    replace = True to accomplish this.
    """
    
    # create path if needed
    if not os.path.exists(ts_dir):
        os.mkdir(os.path.join(ts_dir))
                 
    ## Load in files.
    ts_files = [os.path.join(TS_DIR,file) for file in os.listdir(TS_DIR) if re.search('\.csv',file)]             
    if (len(ts_files) > 0) & (replace == False):
        if verbose:
            print('Ingesting timeseries data from files.')
        ts = {re.sub('.csv','',os.path.split(file)[1][3:]) : pd.read_csv(file) for file in ts_files}
        return(ts)

    df = load_avvail_data(data_dir)

    ## Save all invoices from DF
    # df.to_csv(os.path.join(TS_DIR,'all-invoices.csv'),index = False)                                 

    ## Determine top 10 countries wrt revenue.
    top10_countries = df.groupby('country', as_index = False).\
    agg({'price':'sum'}).sort_values('price', ascending = False).country[:10]       
            
    # Convert the data into timeseries.
    ts = {}

    for country in top10_countries:
        ts[country] = timeseries_aggregate(df, country = country)
        ts[country].to_csv(os.path.join(TS_DIR, f'ts_{country}.csv'), index = False)
    
    ts['all'] = timeseries_aggregate(df, country = None)
    ts['all'].to_csv(os.path.join(TS_DIR, f'ts_all.csv'), index = False)
    
    return(ts)
    


def engineer_features(dataframe, training = True):
    """
    Engineer features as to prepare them for modelling. Our Goal is to model the
    next 30 days of revenue, so for any given day that will be the target.
    """
    
    if training:
        dataframe = dataframe.head(dataframe.shape[0] - 30)
    
    eng_df = pd.DataFrame({})
    
    #Engineer the target.
    eng_df['target'] = [dataframe.daily_revenue[i:i+30].sum() for i in range(dataframe.shape[0])]

    #Engineer the features.
    past_intervals = [7,14,30,60, 365]
    for interval in past_intervals:
         eng_df[f'revenue_{interval}d'] = dataframe.rolling(interval, min_periods = 1)['daily_revenue'].sum()
        
    #Add some non-revenue features
    eng_df[f'views_{30}d'] = dataframe.rolling(30, min_periods = 1)['daily_views'].sum()
    eng_df[f'purchases_{30}d'] = dataframe.rolling(30, min_periods = 1)['daily_purchases'].sum()
    
    
    dates = dataframe.dates
    eng_df['dates'] = dates
    
    return eng_df


# Main Funciton 
if __name__ == "__main__":
    
    run_start = time.time()
    print ("start loading data")
  
    ## ingest data
    ts = load_ts(TS_DIR, DATA_DIR, replace = True)
    
    ## 
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...run time:", "%d:%02d:%02d"%(h, m, s))
    
    print("done")
