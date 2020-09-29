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
    return [sum(map(lambda x: x.get('length'), sites)) for sites in df['sites']]


def utc_to_tz(datetime_object, location='GMT'):
    return datetime_object.replace(tzinfo=timezone.utc).astimezone(tz=tz.gettz(location))


def get_datetime(date, time):
    return datetime.strptime(str(date.date()) + ' ' + time, '%Y-%m-%d %H:%M:%S')


def datetime_to_hours(datetime_object):
    return datetime_object.hour + datetime_object.minute / 60


def extract_hour_local(df):
    return list(df.apply(lambda row: datetime_to_hours(utc_to_tz(get_datetime(row.date, row.time),
                        location_to_tzinfo.get(row.location, row.location)).time()), axis=1))

def extract_hour(df):
    return [int(time.split(':')[0]) for time in df['time']]


def is_inside_interval(number, interval):
    return (number >= interval[0]) & (number <= interval[1])


def is_inside_any_intervals(number, intervals):
    return any(is_inside_interval(number, interval) for interval in intervals)


def extract_hour_joe(hours, joe_hours = ((11, 14), (20, 23))):
    return [int(is_inside_any_intervals(hour, joe_hours)) for hour in hours]


def is_inside_interval(number, interval):
    return (number >= interval[0]) & (number <= interval[1])


def is_inside_any_intervals(number, intervals):
    return any(is_inside_interval(number, interval) for interval in intervals)


def intersection_ratio(set_this, set_reference):
    return len(set(set_this) & set(set_reference)) / len(set(set_this)) if len(set(set_this)) > 0 else 0


def extract_sites_ratio(df, sites_joe_list):
    return [intersection_ratio([site.get('site') for site in sites], sites_joe_list)
                       for sites in df['sites']]


def extract_site_old(df, sites_joe_list):
    return [not {site.get('site') for site in sites}.isdisjoint(sites_joe_list)
                       for sites in df['sites']]


def extract_lengths(df, sites_joe_list):
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
    
    df[features] = df[features].astype('category')

    le = {feat: LabelEncoder().fit(df[feat]) for feat in features}
        
    return df, le


def encode_features(df, features, le):

    for feat in features:
        df[feat] = le[feat].transform(df[feat])
        
    return df


def encode_joe(is_joe_bool_list, encode_dict={True: user_id_joe, False: 1}):
    return [encode_dict[is_joe] for is_joe in is_joe_bool_list]


def transform_features(df, features, features_categorical, joe_top_sites, le):
    """Adds new dependant features, removes some unused ones and convert others to categorical."""

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
    # df[joe_top_sites] = scaler.transform(df[joe_top_sites])
    
    
    return df[features]


def print_scores(y_pred, y_true):
    print('{0:.4%}'.format(accuracy_score(y_pred, y_true)), 'is the accuracy of the classifier.')
    print('{0:.4%}'.format(f1_score(y_pred, y_true, pos_label=user_id_joe)), 'is the F1-score of the classifier.')
    print('{0:.2%}'.format(recall_score(y_pred, y_true, pos_label=user_id_joe)), 'of the Joe\'s accesses are detected.')
    print('{0:.2%}'.format(precision_score(y_pred, y_true, pos_label=user_id_joe)), 'of the detections are truly from Joe.')



def main(
    file_json = './data/verify.json',
    file_pickle = 'catchjoe.pickle',
    file_output = 'catch_joe_output.txt'
    ):

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
    main()