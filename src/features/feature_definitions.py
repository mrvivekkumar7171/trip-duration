import pathlib, sys
import pandas as pd
import numpy as np
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from logger import infologger


def haversine_array(lat1, lng1, lat2, lng2):
    """Calculate the great circle distance in kilometers between two points on the Earth specified in decimal degrees."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    """Calculate the Manhattan distance in kilometers between two points."""
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    """Calculate the bearing in degrees between two points on the Earth specified in decimal degrees."""
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def datetime_feature_fix(df):
    """Fix the datetime features in the DataFrame."""
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
    df.loc[:, 'pickup_date'] = df['pickup_datetime'].dt.date
    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)

def create_dist_features(df):
    """Create distance features in the DataFrame."""
    df.loc[:, 'distance_haversine'] = haversine_array(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    df.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    df.loc[:, 'direction'] = bearing_array(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    

def create_datetime_features(df):
    """Create datetime features in the DataFrame."""
    df.loc[:, 'pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df.loc[:, 'pickup_hour'] = df['pickup_datetime'].dt.hour
    df.loc[:, 'pickup_minute'] = df['pickup_datetime'].dt.minute
    df.loc[:, 'pickup_dt'] = (df['pickup_datetime'] - df['pickup_datetime'].min()).dt.total_seconds()
    df.loc[:, 'pickup_week_hour'] = df['pickup_weekday'] * 24 + df['pickup_hour']

def feature_build(df, tag):
    """Building necessary features required for prediction from a given DataFrame"""
    try:
        datetime_feature_fix(df)
        create_dist_features(df)
        create_datetime_features(df)
        do_not_use_for_training = ['id', 'pickup_datetime', 'dropoff_datetime', 'check_trip_duration', 'pickup_date', 'pickup_datetime_group']
        feature_names = [f for f in df.columns if f not in do_not_use_for_training]
    except Exception as e:
        infologger.info(f'Feature build has been failed with error : {e}')
    else:
        infologger.info(f'Features created successfully, We have {len(feature_names)} features in {tag}.')
        return df[feature_names]

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_path = home_dir / 'data' / 'raw' / 'test.csv'
    data_path_str = data_path.as_posix()

    data = pd.read_csv(data_path_str, nrows=10)
    feature_build(data, 'test')
    print(data.head())