import json
from pymongo import MongoClient

def create_db():
	client = MongoClient()
	# Access/Initiate Database
	db = client['fraud_db']
	# Access/Initiate Table
	tab = db['events']
	with open('data/data.json') as f:
	   file_data = json.load(f)
	tab.insert(file_data)


def get_data():
   client_name = 'fraud_db'
   tab_name = 'events'
   mongo_cols = {'acct_type','user_type','email_domain','venue_state','venue_name'}
   client = MongoClient()
   db = client[client_name]
   tab = db[tab_name]
   cursor = tab.find(None,mongo_cols)
   df = pd.DataFrame(list(cursor))




