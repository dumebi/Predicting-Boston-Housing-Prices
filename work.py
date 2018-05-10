import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)

def load_data():
    work_data = pd.read_csv('AUDJPY_2.csv')
    print "work dataset has {} data points with {} variables each.".format(*work_data.shape)
    # print housing_data.dtypes
    return work_data


def get_features(data):
    features = data.drop(['Date', 'Time', 'Close', 'High', 'Low', 'Total Ticks'], axis = 1)
    #features['timestamp'] = pd.to_datetime(features['timestamp'])
    #features = features.set_index('timestamp')
    return features

def get_close(data):
    close = data.drop(['Date', 'Time', 'Open', 'High', 'Low', 'Total Ticks'], axis = 1)
    return close

def predict(open):
    data = load_data()
    features = get_features(data)
    close = get_close(data)
    print features.head(10)
    print close.shape

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(features, close, test_size=0.4, shuffle=True, random_state=101)
    regressor.fit(X_train, y_train)

    #new_timestamp = pd.to_datetime(timestamp)
    test = np.array([[open]])
    pred = regressor.predict(test)
    #return pred
    if(pred > open):
        return "up"
    else:
        return "down"


# Fit the data
pred =  predict(85.226)
print pred
# print r2_score(y_test, pred)

