
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import time
import re


#Import functions from data_ingestion script
from project_setup import PROJECT_DATA_DIR, DATA_DIR, IMAGE_DIR

from data_ingestion import load_avvail_data

def save_fig(figure_id):
    """
    Save Images as pdf.
    """
    
    #Check Directory
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        
    image_path = os.path.join(IMAGE_DIR, figure_id + '.png')
    plt.savefig(image_path, format = 'png')
    



def create_plots(df):
    """
    Generate plots
    """
    sns.set_style('darkgrid')
    
    
    #Total revenues over time.
    fig, ax = plt.subplots(figsize=(20, 10)) 
    
    df1 = df.groupby(['year_month'], as_index = False).agg({'price':'sum'})
    df1.plot(x = 'year_month', y = 'price', ax = ax)
    plt.xlabel('Dates')
    plt.ylabel('Total Revenue')
    plt.title('Total Revenue over Time')
    save_fig('Revenue over Time')
    
    
    #Total revenues for top 10 countries
    fig, ax = plt.subplots(figsize=(20, 10)) 
    top_countries = [country for country in df.groupby('country', as_index = False).agg({'price':'sum'}).\
                sort_values('price', ascending = False).head(10).country]
    df2 = df[np.isin(df.country,top_countries)]
    df2 = df2.groupby(['year_month', 'country'], as_index = False).agg({'price' : 'sum'})
    sns.lineplot(x = df2.year_month, y = df2.price, hue = df2.country)
    plt.xlabel('Dates')
    plt.ylabel('Revenue')
    plt.title('Total Revenue over Time (Top 10 Countries)')
    save_fig('Revenue for the Top 10 Countries')
    
    
    #Total revenues for top 10 countries (minus United Kingdom)
    fig, ax = plt.subplots(figsize=(20, 10)) 
    top_countries = [country for country in df.groupby('country', as_index = False).agg({'price':'sum'}).\
                sort_values('price', ascending = False).head(10).country]
    df2 = df[np.isin(df.country,top_countries)]
    df2 = df2.groupby(['year_month', 'country'], as_index = False).agg({'price' : 'sum'})
    df2 = df2[df2.country != 'United Kingdom']
    sns.lineplot(x = df2.year_month, y = df2.price, hue = df2.country)
    plt.xlabel('Dates')
    plt.ylabel('Revenue')
    plt.title('Top Country Revenue (Minus United Kingdom)')
    save_fig('Top Country Revenue (Minus United Kingdom)')
    
    
    
    #Total Revenues per year.
    fig, ax = plt.subplots(figsize=(20, 10)) 
    df3 = df.groupby('year', as_index = False).agg({'price':'sum'})
    sns.barplot(x = df3.year, y = df3.price)
    plt.xlabel('Years')
    plt.ylabel('Revenue')
    plt.title('Total Revenue by Year')
    save_fig('Revenue by Year')
    
    
    
if __name__ == "__main__":
    
    run_start = time.time()
    print ("start loading data")
    
    df = load_avvail_data(DATA_DIR)

    print ("create necessary plots")
    create_plots(df)
    
    print("METADATA")
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...run time:", "%d:%02d:%02d"%(h, m, s))
    
    print("done")    
