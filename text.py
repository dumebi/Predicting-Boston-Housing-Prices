
import numpy as np
import pandas as pd


def load_data():
    '''Load the Boston dataset.'''
    housing_data = pd.read_csv('new_house.csv')
    housing_data = housing_data.loc[~housing_data['Price'].isnull()]
    housing_data = housing_data.tail(100000)
    print "Boston housing dataset has {} data points with {} variables each.".format(*housing_data.shape)
    # print housing_data.dtypes
    return housing_data

def load_predict_data():
    '''Load the Boston dataset.'''
    housing_data = pd.read_csv('new_house.csv')
    housing_data = housing_data.loc[housing_data['Price'].isnull()]
    print "Boston housing dataset has {} data points with {} variables each.".format(*housing_data.shape)
    # print housing_data.dtypes
    return housing_data

def get_features(data):
    features = data.drop(['Price','ID','Postcode','Street','Locality','Date','Town','District','County'], axis = 1)

    features.Property_Type = pd.Categorical(features.Property_Type.astype('category').cat.codes).astype(int)
    features.Old_New = pd.Categorical(features.Old_New.astype('category').cat.codes).astype(int)
    features.Duration = pd.Categorical(features.Duration.astype('category').cat.codes).astype(int)
    # features.Town = pd.Categorical(features.Town.astype('category').cat.codes).astype(int)
    # features.District = pd.Categorical(features.District.astype('category').cat.codes).astype(int)
    # features.County = pd.Categorical(features.County.astype('category').cat.codes).astype(int)
    features.PPD_Category_Type = pd.Categorical(features.PPD_Category_Type.astype('category').cat.codes).astype(int)

    return features

def get_prices(data):
    prices = data['Price'].astype(int)
    return prices


def price_statistics(prices):
    # TODO: Minimum price of the data
    minimum_price = np.min(prices)

    # TODO: Maximum price of the data
    maximum_price = np.max(prices)

    # TODO: Mean price of the data
    mean_price = np.mean(prices)

    # TODO: Median price of the data
    median_price = np.median(prices)

    # TODO: Standard deviation of prices of the data
    std_price = np.std(prices)

    # Show the calculated statistics
    print "Statistics for Boston housing dataset:\n"
    print "Minimum price: ${:,.2f}".format(minimum_price)
    print "Maximum price: ${:,.2f}".format(maximum_price)
    print "Mean price: ${:,.2f}".format(mean_price)
    print "Median price ${:,.2f}".format(median_price)
    print "Standard deviation of prices: ${:,.2f}".format(std_price)
    print "\n"

data = load_data()
predict_data  = load_predict_data()
predict_features = get_features(predict_data)
prices = get_prices(data)
features = get_features(data)
price_statistics(prices)

from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.5,  random_state=0)

# Fit the data
regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)
for i, price in enumerate(regressor.predict(predict_features)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)
# print pred
print r2_score(y_test, pred)



