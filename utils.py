import pandas as pd
import numpy as np
import math

import os
from path import Path
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


# ======================================== #
#          Some useful constants
# ======================================== #

WEEKDAYS = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}


# ======================================== #
#          Some useful functions
# ======================================== #

# Functions related to the time and data

def unix_2_realtime(date_unix):
    unix_time = date_unix
    real_time = datetime.datetime.fromtimestamp(unix_time)
    return real_time.strftime("%Y-%m-%d %H:%M:%S")


def target_tz(x, tz_name):
    return pd.to_datetime(x, unit='s',utc=True).tz_convert(tz_name)


def get_weekday(x, form='%Y-%m-%d %H:%M:%S'):
    date_formatted = datetime.strptime(x, form)
    return date_formatted.weekday()


def get_hour(x, form='%Y-%m-%d %H:%M:%S'):
    date_formatted = datetime.strptime(x, form)
    return date_formatted.hour


# Get a grid from a data set

def get_bbox_and_size(data, n_rows=80, n_cols=80):
    assert type(data) == pd.core.frame.DataFrame
    
    min_lon = min(data['longitude'])
    max_lon = max(data['longitude'])
    min_lat = min(data['latitude'])
    max_lat = max(data['latitude'])

    row_size = abs(max_lat - min_lat) / (n_rows-1)
    col_size = abs(max_lon - min_lon) / (n_cols-1)
    
    print('col_size', col_size)
    print('row_size', row_size)
    
    return {'min_longitude': min_lon, 'max_longitude':max_lon, 
            'min_latitude' : min_lat, 'max_latitude':max_lat, 
            'row_size': row_size, 'col_size':col_size}
    

def get_passenger_grid_dict(data, n_rows=80, n_cols=80, min_lon=None, max_lon=None, min_lat=None, max_lat=None):
    """
    Gets the grid in the format of a dictionary whose keys are the cell (i, j) and the value
    is a list of all the points in that particular cell.
    """
    assert type(data) == pd.core.frame.DataFrame
    
    grid = {}
    for r in range(n_rows):
        for c in range(n_cols):
            grid[(r, c)] = []
        
    if min_lon is None or max_long is None or min_lat is None or max_lat is None:
        min_lon = min(data['longitude'])
        max_lon = max(data['longitude'])
        min_lat = min(data['latitude'])
        max_lat = max(data['latitude'])
        
        row_size = (max_lat - min_lat) / (n_rows-1)
        col_size = (max_lon - min_lon) / (n_cols-1)
        
    for i in range(data.shape[0]):
        if data.iloc[i]['in_out'] == 'in':
            lat = data.iloc[i]['latitude']
            lon = data.iloc[i]['longitude']
            r_index = int((lat - min_lat)/row_size)
            c_index = int((lon - min_lon)/col_size)
        
            grid[(r_index, c_index)].append(i)

    return grid


def get_passenger_grid(data, n_rows=80, n_cols=80, min_lon=None, max_lon=None, min_lat=None, max_lat=None):
    """
    Gets the grid in the format of a 2D Matrix where each cell shows the number of data samples in
    that cell.
    """
    assert type(data) == pd.core.frame.DataFrame
    
    grid = [[0 for c in range(n_cols)] for r in range(n_rows)]
        
    if min_lon is None or max_lon is None or min_lat is None or max_lat is None:
        min_lon = min(data['longitude'])
        max_lon = max(data['longitude'])
        min_lat = min(data['latitude'])
        max_lat = max(data['latitude'])
        
    row_size = (max_lat - min_lat) / (n_rows-1)
    col_size = (max_lon - min_lon) / (n_cols-1)
        
    for i in range(data.shape[0]):
        if data.iloc[i]['in_out'] == 'in':
            lat = data.iloc[i]['latitude']
            lon = data.iloc[i]['longitude']
            r_index = int((lat - min_lat)/row_size)
            c_index = int((lon - min_lon)/col_size)

            grid[r_index][c_index]+=1

    return grid


def get_empty_taxi_grid(data, n_rows=80, n_cols=80, min_lon=None, max_lon=None, min_lat=None, max_lat=None):
    """
    Gets the grid in the format of a 2D Matrix where each cell shows the number of data samples in
    that cell where the taxi is empty.
    """
    assert type(data) == pd.core.frame.DataFrame
    
    grid = [[0 for c in range(n_cols)] for r in range(n_rows)]
        
    if min_lon is None or max_long is None or min_lat is None or max_lat is None:
        min_lon = min(data['longitude'])
        max_lon = max(data['longitude'])
        min_lat = min(data['latitude'])
        max_lat = max(data['latitude'])
        
        row_size = (max_lat - min_lat) / (n_rows-1)
        col_size = (max_lon - min_lon) / (n_cols-1)
        
    for i in range(data.shape[0]):
        if data.iloc[i]['occupancy'] == 0:
            lat = data.iloc[i]['latitude']
            lon = data.iloc[i]['longitude']
            r_index = int((lat - min_lat)/row_size)
            c_index = int((lon - min_lon)/col_size)

            grid[r_index][c_index]+=1

    return grid


# Functions related to the distance

def get_distance(pt_1, pt_2):
    """
    Assuming, pt_1 is formatted as (longitude_1, latitude,_1) and pt_2 is formatted as 
    (longitude_2, latitude,_2), measure the distance between them in Km.
    To recall, the distance in Km can be found by multiplying the difference by 111.139
    """
    #     return np.sqrt((pt_1[0] - pt_2[0])**2 * 111.139 + (pt_1[1]-pt_2[1])**2 * 111.139) 
    return haversine((pt_1[1], pt_1[0]), (pt_2[1], pt_2[0]))
    
    
def get_empty_taxis_around(df, weekday, hour, longitude, latitude, distance=3):
    all_data = df[(df['day']==weekday) & (df['hour']==hour)]
    all_data.reset_index(drop=True, inplace=True)
    
    # Obtain the probability (ratio / statistical probability) of empty taxis around
    count_taxis_around = 0
    count_empty_around = 0
    
    for i in range(all_data.shape[0]):
        lon_i, lat_i = all_data.iloc[i]['longitude'], all_data.iloc[i]['latitude']
        if get_distance((lon_i, lat_i), (longitude, latitude)) <= distance:
            count_taxis_around += 1
            if all_data.iloc[i]['occupancy']==0:
                count_empty_around += 1
            
    return count_taxis_around, count_empty_around


def cluster_dataframe(df, eps=0.01, min_samples=5):
    """
    To recall, we made it so that each cell is about 1 Km square, i.e., the cell dimensions are
    roughly 0.0091 * 0.0091. We set the default value for DBSCAN's epsilon parameter to be roughly
    the same (0.01) so that taxis within 1 Km of one another are considered in the same cluster.
    We also assume that the dataframe is pre-filtered, so that we use the data directly.
    """
    
    X = df[["longitude", "latitude"]].to_numpy()
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    
    return X, clustering
    

def plot_clusters(X, labels, n_clusters, n_noise, clusterer):
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clusterer.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters}")
    plt.show()

