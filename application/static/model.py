import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import json
from pymongo import MongoClient
import pickle
from src.cleaner import Cleaner
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class GBModel():
	def __init__(self):
		# self.model = GradientBoostingClassifier(learning_rate=0.01, max_depth=8, 
		# 	max_features=5, min_samples_leaf=5, n_estimators=1500)
		self.model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=12, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=2500, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
		pass
	
	def fit(self, X, y):
		# Import X and y as numpy arrays
		self.X = X
		self.y = y
		self.model.fit(self.X, self.y)
		filename = 'data/model.pkl'
		pickle.dump(self, open(filename, 'wb')) #self.model?
		self.cols = list(self.X.columns)
		return self
	
	def predict(self, X):
		predictions = self.model.predict(X)
		return predictions

	def score(self, X, y):
		score = self.model.score(X, y)
		return score


	def importance(self, colnames):
		feature_importances = self.model.feature_importances_
		top10_colindex = np.argsort(feature_importances)[::-1][0:10]
		feature_importances = feature_importances[top10_colindex]
		feature_importances = feature_importances / np.sum(feature_importances)
		y_ind = np.arange(9, -1, -1) # 9 to 0
		fig = plt.figure(figsize=(8, 8))
		plt.barh(y_ind, feature_importances, height = 0.3, align='center')
		plt.ylim(y_ind.min() + 0.5, y_ind.max() + 0.5)
		plt.yticks(y_ind, [colnames[i] for i in top10_colindex])
		plt.xlabel('Relative feature importances')
		plt.ylabel('Features')
		plt.show()
	
	def plot_partial_dependencies(self, colnames):
		feature_importances = self.model.feature_importances_
		top10_colindex = np.argsort(feature_importances)[::-1][0:10]
		#fig, axs = plt.subplots(5,2, figsize=(20,20))
		fig, axs = plot_partial_dependence(self.model, self.X, features=top10_colindex, 
			feature_names = colnames, figsize=(20,20), grid_resolution=100)

		fig.set_figwidth(20)
		fig.set_figheight(20)
		fig.tight_layout()
		#plt.figure(figsize=(5,5))
		plt.show()

def get_data():
   client_name = 'fraud_db'
   tab_name = 'events'
   #mongo_cols = {'acct_type','user_type','email_domain','venue_state','venue_name'}
   client = MongoClient()
   db = client[client_name]
   tab = db[tab_name]
   cursor = tab.find(None) #mongo_cols)
   df = pd.DataFrame(list(cursor))
   return df


if __name__ == '__main__':
	# read data
	dataframe = get_data()
	#print(dataframe)
	#clean data
	y = dataframe['acct_type'].str.contains('fraud').astype(int)
	X_train, X_test, y_train, y_test = \
        train_test_split(dataframe, y, random_state = 142)

	print('cleaning....')
	clean = Cleaner()
	clean.fit(X_train)
	X_train = clean.transform(X_train)
	X_test = clean.transform(X_test)


	print('Fitting....')
	#fit model
	gb = GBModel()
	gb.fit(X_train, y_train)
	
	print('score: {}'.format(gb.score(X_test,y_test)))


	



