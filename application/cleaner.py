import pandas as pd
import numpy as np
import pickle


class Cleaner(object):
    #this will take a raw pandas DF and prepare it for fitting/prediction
    def __init__(self):

        self.risk_dicts = pickle.load(open('risk_dict.pkl', 'rb'))

    def get_payments_info(self, x):
        #takes the payment info column and returns the number of payouts made to the user and
        # the number of distinct payees
        temp_df = pd.DataFrame(x)
        if len(temp_df) ==0:
            return [0,0]
        else:
            return [temp_df['amount'].mean(),len(temp_df['address'].unique())]

    def get_ticket_info(self,x):
        #takes the ticket_info column from the training data to return if the tickets are val_available
        # for sale, how many have been sold, the total value available and the most expensive ticket price
        temp_df = pd.DataFrame(x)
        if len(temp_df) ==0:
            return [0,0,0,0]
        else:
            df_cols = list(temp_df.columns)
            if 'availability' in df_cols:
                available = temp_df['availability'].max()
            else:
                available = 0
            if 'cost' in df_cols:
                if 'quantity_sold' in df_cols:
                    sold = np.sum(temp_df['cost'].values*temp_df['quantity_sold'].values)
                else:
                    sold = 0
                if 'quantity_total' in df_cols:
                    val_available = np.sum(temp_df['cost'].values*temp_df['quantity_total'].values)
                else:
                    val_available = 0
                max_price = temp_df['cost'].max()
            else:
                sold = 0
                val_available = 0
                max_price = 0
        return [available,sold,val_available,max_price]

    def transform(self, df):
        #input requires a Pandas dataFrame containing the raw data and outputs a cleaned and transformed
        #version ready for training/predicting
        X = df[['object_id','org_facebook','org_twitter']].copy(deep = True)
        X['org_facebook'].fillna(0,inplace = True)
        X['org_twitter'].fillna(0,inplace = True)
        df_cols = list(df.columns)
        key = 'venue_address'
        X['{}_risk'.format(key)]=df[key].map(self.risk_dicts[key])
        X['{}_risk'.format(key)].fillna(3,inplace = True)
        for key in self.risk_dicts.keys():
            X['{}_risk'.format(key)]=df[key].map(self.risk_dicts[key])
            X['{}_risk'.format(key)].fillna(0,inplace = True)

        plain_cols = ['body_length','channels','gts','fb_published','has_analytics','has_header',
                        'has_logo','name_length','num_order','num_payouts','sale_duration','sale_duration2',
                        'show_map','user_age']
        for col_name in plain_cols:
            if col_name in df_cols:
                X[col_name] = df[col_name].fillna(0)
            else:
                X[col_name] = 0

        X['listed'] = df['listed'].map({'y':1,'n':0}).fillna(0)

        payout_info = df['previous_payouts'].apply(lambda x: self.get_payments_info(x))
        X['avg_payout']=payout_info.apply(lambda x: x[0])
        X['num_payees']=payout_info.apply(lambda x: x[1])

        ticket_info = df['ticket_types'].apply(lambda x: self.get_ticket_info(x))
        X['tickets_available']=ticket_info.apply(lambda x: x[0])
        X['value_sold']=ticket_info.apply(lambda x: x[1])
        X['value_available']=ticket_info.apply(lambda x: x[2])
        X['max_price']=ticket_info.apply(lambda x: x[3])
        return X.set_index('object_id')
