import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def barplotter(df, column):
	newdf = pd.merge(df[column].value_counts().reset_index(),
         df[column][df.fraud == 1].value_counts().reset_index(), 
         how='left', on='index')

	newdf.columns = ['category', 'num_occurences', 'num_fraud']
	newdf['freq_fraud'] = newdf.num_fraud / newdf.num_occurences
	newdf.fillna(0, inplace=True)

	ax = sns.barplot(newdf.category, newdf.freq_fraud)
	ax.set(xlabel=column, ylabel='percent fraud')
	
	labels = [str(txt) for txt in newdf.num_occurences]
	rects = ax.patches
	for rect, label in zip(rects, labels):
	    height = rect.get_height()
	    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
	            ha='center', va='bottom')


def cleaner(df):
    df['fraud'] = df.acct_type.str.contains('fraud').astype(int)
    #df['spam'] = df.acct_type.str.contains('spam').astype(int)
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
    df['listed'] = df.listed.map({"y":1, "n":0})
    ###
    df["user_created"] = pd.to_datetime(df["user_created"], unit='s')
    ###
    #df['sale_duration'] = df['sale_duration'].astype(int)
    return df

def stage_score_plot(estimator, X_train, y_train, X_test, y_test):
    '''
    Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                X_train: 2d numpy array
                y_train: 1d numpy array
                X_test: 2d numpy array
                y_test: 1d numpy array

    Returns: A plot of the number of iterations vs the MSE for the model for
    both the training set and test set.
    '''
    mse_train = np.zeros(estimator.n_estimators)
    mse_test = np.zeros(estimator.n_estimators)
    estimator.fit(X_train,y_train)
    for ind, (yh_test,yh_train) in enumerate(zip(estimator.staged_predict(X_test),
                                                 estimator.staged_predict(X_train))):
        mse_test[ind]=np.sum(yh_test!=y_test)/len(y_test)
    
    plt.plot(np.r_[0:estimator.n_estimators],mse_test,
             label ='{} Test Rate {}'.format(estimator.__class__.__name__, estimator.learning_rate))
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


