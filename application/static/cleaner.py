import pandas as pd
import numpy as np
import pickle


class Cleaner(object):
    #this will take a raw pandas DF and prepare it for fitting/prediction
    def __init__(self):
        self.risk_dicts = pickle.load(open('static/risk_dict.pkl', 'rb'))

    def transform(self, df):
        X = df[['object_id','org_facebook','org_twitter']].copy(deep = True)
        X['org_facebook'].fillna(0,inplace = True)
        X['org_twitter'].fillna(0,inplace = True)


        for key in self.risk_dicts.keys():
            X['{}_risk'.format(key)]=df[key].map(self.risk_dicts[key])
            X['{}_risk'.format(key)].fillna(0,inplace = True)

        plain_cols = ['body_length','channels','fb_published']
        for col_name in plain_cols:
            X[col_name] = df[col_name].fillna(0)

        return X.set_index('object_id')

def cleaner(df):
    # takes a Dataframe form the raw Mongo DB  request and returns a clean

    df["approx_payout_date"] = pd.to_datetime(df["approx_payout_date"], unit='s')
    ###
    conditions = [
        (df['country'] == 'US'),
        (df['country'] == 'GB'),
        (df['country'] == 'CA'),
        (df['country'].isna() == True)]
    choices = ['High', 'High', 'High', 'N/A']
    df['country_bucket'] = np.select(conditions, choices, default='Low')
    ###
    #df['delivery_method'] = df['delivery_method'].astype(int)
    ###
    df["event_created"] = pd.to_datetime(df["event_created"], unit='s')
    df["event_end"] = pd.to_datetime(df["event_end"], unit='s')
    df["event_published"] = pd.to_datetime(df["event_published"], unit='s')
    df["event_start"] = pd.to_datetime(df["event_start"], unit='s')
    ###
    df['listed'] = np.where(df['listed'] == 'y', 1, 0)
    ###
    df["user_created"] = pd.to_datetime(df["user_created"], unit='s')
    ###
    #df['sale_duration'] = df['sale_duration'].astype(int)
    return df
