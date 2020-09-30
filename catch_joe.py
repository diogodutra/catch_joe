import argparse

from datetime import datetime
from dateutil import tz
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import pickle

from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



user_id_joe = 0


location_to_tzinfo = {
    'Canada/Toronto': 'Canada/Eastern',
    'USA/Chicago': 'America/Chicago',
    'France/Paris': 'Europe/Paris',
}


def extract_duration(df):
    """
    Extracts from dataset the duration of each session.

    Args:
        df (pandas.DataFrame): dataset with history of accesses (ex: pandas.read_json('verify.json'))

    Returns:
        durations (list of float): Summation of all lengths per row in the dataframe
    """
    return [sum(map(lambda x: x.get('length'), sites)) for sites in df['sites']]


def utc_to_tz(datetime_gmt, location='GMT'):
    """
    Calculates the local datetime converting from GMT.

    Args:
        datetime_gmt (datetime): GMT datetime
        location (str, default='GMT'): local timezone to be converted to

    Returns:
        datetime_local (datetime): local datetime
    """
    return datetime_gmt.replace(tzinfo=timezone.utc).astimezone(tz=tz.gettz(location))


def get_datetime(date, time):
    """
    Creates datetime object merging date and time.

    Args:
        date (datetime): date as found in 'date' column from dataset
        time (timestamp): time as found in 'time' column from dataset

    Returns:
        datetime_gmt (datetime): GMT datetime
    """
    return datetime.strptime(str(date.date()) + ' ' + time, '%Y-%m-%d %H:%M:%S')


def datetime_to_hours(datetime_object):
    """
    Calculates the hour of the day including minutes as decimals.

    Args:
        datetime_object (datetime): date and time

    Returns:
        hour (float): hour of the day with minutes as decimals
    """
    return datetime_object.hour + datetime_object.minute / 60


def extract_hour_local(df):
    """
    Extracts from dataset the local hour including minutes as decimals.

    Args:
        df (pandas.DataFrame): dataset with history of accesses (ex: pandas.read_json('verify.json'))

    Returns:
        local_hour (list of float): local hours of the day with minutes as decimals
    """
    return list(df.apply(lambda row: datetime_to_hours(utc_to_tz(get_datetime(row.date, row.time),
                        location_to_tzinfo.get(row.location, row.location)).time()), axis=1))

def extract_hour(df):
    """
    Extracts from dataset the GMT hour (excluding minutes).

    Args:
        df (pandas.DataFrame): dataset with history of accesses (ex: pandas.read_json('verify.json'))

    Returns:
        gmt_hour (list of int): GMT hours of the day (without minutes)
    """
    return [int(time.split(':')[0]) for time in df['time']]


def is_inside_interval(number, interval):
    """
    Checks if number is inside an interval

    Args:
        number (int or float): value to be checked
        interval (tuple of 2 ints or floats): interval to check if number is within

    Returns:
        is_inside (bool): condition if number is inside the interval
    """
    return (number >= interval[0]) & (number <= interval[1])


def is_inside_any_intervals(number, intervals):
    """
    Checks if number is inside any of the intervals

    Args:
        number (int or float): value to be checked
        intervals (list of tuple of 2 ints or floats): intervals to check if number is within

    Returns:
        is_inside (bool): condition if number is inside any of the intervals
    """
    return any(is_inside_interval(number, interval) for interval in intervals)


def extract_hour_joe(hours, joe_hours = ((11, 14), (20, 23))):
    """
    Checks if hours are inside the times that Joe uses to access the internet

    Args:
        hours (list of int or float): hours of the day to be checked
        joe_hours (list of tuple of 2 ints or floats): intervals to check if hours are within

    Returns:
        is_inside (list of bool): conditions if elements of hours are inside any of the intervals
    """
    return [int(is_inside_any_intervals(hour, joe_hours)) for hour in hours]


def intersection_ratio(set_this, set_reference):
    """
    Calculates the intersection ratio between two sets

    Args:
        set_this (set): subset to be have its ratio calculated, ranging from 0 to 1
        set_reference (set): set with values to be compared

    Returns:
        ratio (float): percentage of 'set_this' elements contained in 'set_reference'
    """
    return len(set(set_this) & set(set_reference)) / len(set(set_this)) if len(set(set_this)) > 0 else 0


def extract_sites_ratio(df, sites_joe_list):
    """
    Extracts from dataset the intersection ratios between the sites compared to Joe's

    Args:
        df (pandas.DataFrame): dataset with history of accesses (ex: pandas.read_json('verify.json'))
        sites_joe_list (list of str): most accessed sites by Joe

    Returns:
        ratios (list of float): percentages of sites in every row of dataset contained in 'sites_joe_list'
    """
    return [intersection_ratio([site.get('site') for site in sites], sites_joe_list)
                       for sites in df['sites']]


def extract_site_old(df, sites_joe_list):
    """
    Checks on dataset which rows contains any of the sites accessed by Joe

    Args:
        df (pandas.DataFrame): dataset with history of accesses (ex: pandas.read_json('verify.json'))
        sites_joe_list (list of str): most accessed sites by Joe

    Returns:
        checks (list of bool): conditions if any of the sites in every row of dataset is contained in 'sites_joe_list'
    """
    return [not {site.get('site') for site in sites}.isdisjoint(sites_joe_list)
                       for sites in df['sites']]


def extract_lengths(df, sites_joe_list):
    """
    Extracts from dataset the session length of each site as a separate extended dataset if it is contained in the Joe's list

    Args:
        df (pandas.DataFrame): dataset with history of accesses (ex: pandas.read_json('verify.json'))
        sites_joe_list (list of str): most accessed sites by Joe

    Returns:
        df (pandas.DataFrame): extended dataset with lengths of each session
    """
    sites_joe_length = np.zeros((df.shape[0], len(sites_joe_list)))
    for i_row, sites in enumerate(df['sites']):
        for site in sites:
            try:
                i_site = sites_joe_list.index(site.get('site'))
                sites_joe_length[i_row, i_site] = site.get('length')
            except:
                # site not found in Joe's history
                pass
           
    df_lengths = pd.DataFrame(sites_joe_length, columns=sites_joe_list, index=df.index)
    
    return df_lengths


def categorize(df, features):
    """
    Encodes and converts values in 'features' columns into categorical type

    Args:
        df (pandas.DataFrame): dataset with history of accesses (ex: pandas.read_json('verify.json'))
        features (list of str): subset of columns to be encoded and converted to categorical type

    Returns:
        df (pandas.DataFrame): dataset with categorical features encoded
        le (dict of LabelEncoder): encoder for each categorical feature
    """    
    df[features] = df[features].astype('category')

    le = {feat: LabelEncoder().fit(df[feat]) for feat in features}
        
    return df, le


def encode_features(df, features, le):
    """
    Encodes values in 'features' columns as defined by the LabelEncoders

    Args:
        df (pandas.DataFrame): dataset with history of accesses (ex: pandas.read_json('verify.json'))
        features (list of str): subset of columns to be encoded and converted to categorical type 
        le (dict of LabelEncoder): encoder for each categorical feature

    Returns:
        df (pandas.DataFrame): dataset with categorical features encoded
    """
    for feat in features:
        df[feat] = le[feat].transform(df[feat])
        
    return df


def encode_joe(is_joe_bool_list, encode_dict={True: user_id_joe, False: 1}):
    """
    Encodes list of booleans (is Joe? True/False) into integer (0=Joe/1=not-Joe)

    Args:
        is_joe_bool_list (list of Bool): conditions if row is from user_id = 0 (Joe)
        encode_dict (dict, Optional): conversion from boolean to integer

    Returns:
        encoded (list of bool): encoded list as specified by the project (0=Joe/1=not-Joe)
    """
    return [encode_dict[is_joe] for is_joe in is_joe_bool_list]


def transform_features(df, features, features_categorical, joe_top_sites, le):
    """
    Adds n dependant features, removes some unused ones and convert others to categorical.

    Args:
        df (pandas.DataFrame): dataset with categorical features encoded
        features (list of str): subset of columns to be included in the output dataframe
        features_categorical (list of str): subset of columns to be encoded and converted to categorical type
        le (dict of LabelEncoder): encoder for each categorical feature

        is_joe_bool_list (list of Bool): conditions if row is from user_id = 0 (Joe)
        encode_dict (dict, Optional): conversion from boolean to integer

    Returns:
        encoded (list of bool): encoded list as specified by the project (0=Joe/1=not-Joe)
    """

    # add some features
    new_features = {
        'duration': extract_duration,
        'hour': extract_hour,
        'sites_ratio': extract_sites_ratio,
        'site_old': extract_site_old,
    }
    
    for feat_name, feat_func in new_features.items():
        if feat_name in features: df[feat_name] = feat_func(df)

    # convert features into category type
    df[features_categorical] = df[features_categorical].astype('category')
    df = encode_features(df, features_categorical, le)    
            
    # add Joe's top site lengths as features
    df = df.join(extract_lengths(df, joe_top_sites))
    
    
    return df[features]


def print_scores(y_pred, y_true):
    """
    Prints accuracy, recall, precision and F-1 scores

    Args:
        y_pred (list of int): predictions from classifier
        y_true (list of int): true labels, as per specification (0=Joe/1=not-Joe)
    """
    print('{0:.4%}'.format(accuracy_score(y_pred, y_true)), 'is the accuracy of the classifier.')
    print('{0:.4%}'.format(f1_score(y_pred, y_true, pos_label=user_id_joe)), 'is the F1-score of the classifier.')
    print('{0:.2%}'.format(recall_score(y_pred, y_true, pos_label=user_id_joe)), 'of the Joe\'s accesses are detected.')
    print('{0:.2%}'.format(precision_score(y_pred, y_true, pos_label=user_id_joe)), 'of the detections are truly from Joe.')



def main(
    file_json = 'verify.json',
    file_pickle = './model/catchjoe.pickle',
    file_output = 'catch_joe_output.txt'
    ):
    """
    Runs the standalone script to predict Joe's accesses

    Args:
        file_json (str): path and filename to JSON file containing the dataset
        file_pickle (str): path and filename to PICKLE file containing the parameters for the predictive model
        file_output (str): path and filename to TXT output file containing the predictions (0=Joe/1=not-Joe)
    """

    # load dataset
    with open(file_pickle, 'rb') as f:
        catch_joe_dict = pickle.load(f)

    df = pd.read_json(file_json)

    # extract features
    df = transform_features(
        df,
        catch_joe_dict['features'],
        catch_joe_dict['features_categorical'],
        catch_joe_dict['joe_top_sites'],
        catch_joe_dict['encoder'],
    )

    # predict Joe (0) or not-Joe (1)
    y_pred = catch_joe_dict['model'].predict(df.values)

    # export predictions to txt
    with open(file_output, 'w') as f:
        for y_i in y_pred:
            f.write(str(y_i)+'\n')

    return y_pred


if __name__ == '__main__':

    # parse script parameters
    parser = argparse.ArgumentParser(
        description="Predicts if Joe (user_id==0) accessed the internet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-j", "--file_json", default="verify.json", help="path and filename to JSON input dataset containing user_id, browser, gender, date, time and sites")
    parser.add_argument("-o", "--file_output", default="catch_joe_output.txt", help="path and filename to TXT output that will contain the prediction labels (0=Joe/1=not-Joe)")
    parser.add_argument("-p", "--file_pickle", default="./model/catch_joe.pickle", help="path and filename to PICKLE containing the parameters for the predictive model")
    args = parser.parse_args()
    
    # Create a dictionary of the shell arguments
    kwargs = vars(args)
    
    # run ML pipeline as main script
    main(**kwargs)