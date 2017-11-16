import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def run():
    learning_rate = 0.0001
    #get original dataframe
    mpgdata = pd.read_csv('c:\codepython\Data\mpg.csv')
    #data preprocessing
    mpgdata['horsepower'] = mpgdata["horsepower"].replace(-1, np.mean(mpgdata["horsepower"]))
    mpgdata['years'] = 83 - mpgdata['model_year']
    mpgdata['cylinders'] = mpgdata['cylinders'].astype('category')
    mpgdata['origin'] = mpgdata['origin'].astype('category')
    toCat = ['cylinders', 'origin']
    dummies = pd.get_dummies(mpgdata[toCat], prefix=toCat)
    toCat.extend(['model_year','name'])
    mpgdata.drop(toCat, axis=1, inplace=True)
    mpgdata = pd.concat([mpgdata, dummies], axis=1)
   
    #generate test, train and cv data
    y_labels =  'mpg'
    x_labels = []
    for x in mpgdata.columns:
        if x!= y_labels:
            x_labels.append(x)

    x_mpgdata = mpgdata.loc[:, x_labels]
    y_mpgdata = mpgdata.loc[:, y_labels]

    ## Preparing Train, CV and Test Data data for regression
    #regular split method
    #mpgx_train_cv, mpgx_test, mpgy_train_cv, mpgy_test = train_test_split(mpgdata[:,x_labels],mpgdata[:,y_labels], test_size=0.20, random_state=42)

    testSplitRatio = 0.20
    totalSplit = 5

    #shuffle Split to get Teain-CV and seperate Test Data.
    testSplit = ShuffleSplit(n_splits=2, test_size=0.20, train_size=None, random_state=100)
    
    #Initialize with empty DataFrame
    mpgx_train_cv = mpgx_test = mpgy_train_cv =  mpgy_test = pd.DataFrame() 
    for train_index, test_index in testSplit.split(x_mpgdata):
        mpgx_train_cv, mpgx_test = x_mpgdata.iloc[train_index], x_mpgdata.iloc[test_index]
        mpgy_train_cv, mpgy_test = y_mpgdata.iloc[train_index], y_mpgdata.iloc[test_index]

    mpg_train_cv = pd.concat([mpgx_train_cv, mpgy_train_cv], axis=1)
    mpg_test = pd.concat([mpgx_test, mpgy_test], axis=1)
    
    mpg_train_ar = []
    mpg_cv_ar = []

    k_fold = KFold(n_splits=totalSplit, shuffle=False, random_state=100)
    for tr_ind, ts_index in k_fold.split(mpgx_train_cv):
        mpgx_train, mpgx_cv = mpgx_train_cv.iloc[tr_ind], mpgx_train_cv.iloc[ts_index]
        mpgy_train, mpgy_cv = mpgy_train_cv.iloc[tr_ind], mpgy_train_cv.iloc[ts_index]
        mpg_train_ar.append(pd.concat([mpgx_train, mpgy_train],axis=1))
        mpg_cv_ar.append(pd.concat([mpgx_cv, mpgy_cv],axis=1))

    print(pd.DataFrame(mpg_train_ar[0]).head(1))
    print(pd.DataFrame(mpg_cv_ar[0]).head(1))

if __name__ == '__main__':
    run()

