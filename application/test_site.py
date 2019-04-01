from collections import Counter
import argparse
from flask import Flask, request, render_template
from pymongo import MongoClient
from cleaner import Cleaner
import time
import requests
import pymongo
import pickle
import pandas as pd
import numpy as np
import folium
app = Flask(__name__)




def update_map(df): ## Run this inside the homepage route
    # cursor = db['test_events'].find()
    #cursor = db['results'].find() ##Used for local testing environment
    # df = pd.DataFrame(list(cursor))
    colors = {True: 'red', False: 'blue'}
    df["venue_latitude"] = df["venue_latitude"].fillna(0)
    df["venue_longitude"] = df["venue_longitude"].fillna(0)
    map = folium.Map(location=[30, 0], zoom_start=1,width='80%',height='100%')
    df['fraud'] = df['fraud_probability'] > .5
    df.apply(lambda row: folium.CircleMarker(location=[row["venue_latitude"], row["venue_longitude"]],
                                          radius=10, fill_color=colors[row['fraud']]).add_to(map), axis=1)
    map.save('templates/map.html')
# Form page to submit text
@app.route('/')
def main_page():
    #first get the data that hasnt been predicted
    cursor = tab.find({'fraud_probability':{'$exists':False}})
    if cursor.count()>0:
        df = pd.DataFrame(list(cursor))
        X = clnr.transform(df)
        pred = fraud_model.predict_proba(X.values)[:,1]
        New_data = pd.DataFrame(np.array([df['object_id'].values,pred,X['value_available'].values]).T)
        for row in New_data.values:
            tab.update_one({'object_id':int(row[0])},{'$set':{'fraud_probability':row[1],
                    'value_available':row[2]}})
#to do: raise the cutoff to .5 when we have the second page up
    cursor = tab.find({'fraud_probability':{'$gt':0.5},'cleared':{'$exists':False}})
    if cursor.count()>0:
        fraud_df = pd.DataFrame(list(cursor))
        display_df = fraud_df[['object_id','name','org_name','fraud_probability','value_available']]
        display_df.columns = ['Event ID','Event Name','Organization','Fraud Prediction','Value Listed']
        conditions = [(display_df['Fraud Prediction'] >= .5), \
                    ((display_df['Fraud Prediction'] < .5) & (display_df['Fraud Prediction'] > .2)), \
                    (display_df['Fraud Prediction'] <=.2)]
        choices = ['high risk','medium risk','low risk']
        display_df['Risk Level'] = np.select(conditions,choices)
        tbl =display_df.to_html(index = False,index_names = False,formatters={'Fraud Prediction':pct_format,
                        'Event ID':id_format}, justify = 'center')
        fraud_events = display_df[display_df['Fraud Prediction'] > .90]

        fraud_count = len(fraud_events)
        total_count = len(display_df)
        percent_fraud = np.around(float(fraud_count) / total_count,decimals=2) *100
    else:
        tbl = 'No Fraudulent Events To Review at this time.'
        fraud_count = 0
        total_count = 0
        percent_fraud = 0



    cursor = tab.find({'cleared':{'$exists':False}})
    if cursor.count()>0:
        fraud_df = pd.DataFrame(list(cursor))
        update_map(fraud_df)


    return render_template(site, fraud_table = tbl, fraud_count=fraud_count, total_count=total_count,percent_fraud=percent_fraud)

# TO DO: separate page which has a full list of events, while the main page has only high-risk events.
@app.route('/full_list')
def full_list():

    cursor = tab.find({'fraud_probability':{'$gt':0.0},'cleared':{'$exists':False}})
    if cursor.count()>0:
        fraud_df = pd.DataFrame(list(cursor))
        display_df = fraud_df[['object_id','name','org_name','fraud_probability','value_available']]
        display_df.columns = ['Event ID','Event Name','Organization','Fraud Prediction','Value Listed']
        conditions = [(display_df.loc[:,'Fraud Prediction'] >= .5), \
                    ((display_df.loc[:,'Fraud Prediction'] < .5) & (display_df.loc[:,'Fraud Prediction'] > .2)), \
                    (display_df.loc[:,'Fraud Prediction'] <=.2)]
        choices = ['high risk','medium risk','low risk']
        display_df['Risk Level'] = np.select(conditions,choices)
        tbl =display_df.to_html(index = False,index_names = False,formatters={'Fraud Prediction':pct_format,
                        'Event ID':id_format}, justify = 'center')
        fraud_events = display_df[display_df['Fraud Prediction'] > .90]

        fraud_count = len(fraud_events)
        total_count = len(display_df)
        percent_fraud = np.around(float(fraud_count) / total_count,decimals=4) *100
    else:
        tbl = 'No Fraudulent Events To Review at this time.'
        fraud_count = 0
        total_count = 0
        percent_fraud = 0



    cursor = tab.find({'cleared':{'$exists':False}})
    if cursor.count()>0:
        fraud_df = pd.DataFrame(list(cursor))
        update_map(fraud_df)


        return render_template('full_list.html', fraud_table = tbl, fraud_count=fraud_count, total_count=total_count,percent_fraud=percent_fraud)



if __name__ == '__main__':

    site = 'index2.html'

    with open('fraud_model.pkl', 'rb') as f:
        fraud_model = pickle.load(f)
    clnr = Cleaner()

    client = MongoClient()
    # Access/Initiate Database
    db = client['fraud_db']
    # Access/Initiate Table
    tab = db['test_events']

    pct_format = lambda x: '{:.0%}'.format(x)
    id_format = lambda x: '{}'.format(x)


    def make_map(): ## Run this inside the homepage route

        map = folium.Map(location=[30, 0], zoom_start=1,width='100%',height='35%')
        map.save('templates/map.html')

    make_map()


    app.run(host='0.0.0.0', port=8000, debug=True)
