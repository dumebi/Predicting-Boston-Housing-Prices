"""
Loading the boston dataset and examining its target (label) distribution.
"""

# Load libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import displays as display
from sklearn.model_selection import  ShuffleSplit, train_test_split


def load_data():
    '''Load the Boston dataset.'''
    housing_data = pd.read_csv('housing.csv')
    print "Boston housing dataset has {} data points with {} variables each.".format(*housing_data.shape)
    return housing_data


def get_features(data):
    features = data.drop('MEDV', axis = 1)

    return features

def get_prices(data):
    prices = data['MEDV']

    #TODO: Inflation rate is 275%
    #prices = np.dot(prices, 275)
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
prices = get_prices(data)
features = get_features(data)
price_statistics(prices)

## Question 1 - Feature Observation
print "Question 1 - Feature Observation:\n"

# 'RM' is the average number of rooms among homes in the neighborhood.
# 'LSTAT' is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
# 'PTRATIO' is the ratio of students to teachers in primary and secondary schools in the neighborhood.

# Answer

# Increase in 'RM' will lead to an increase in MEDV
print "Increase in 'RM' will lead to an increase in MEDV"
# Increase in 'LSTAT' will lead to a decrease in MEDV
print "Increase in 'LSTAT' will lead to a decrease in MEDV"
# Increase in 'PTRATIO' will lead to a decrease in MEDV
print "Increase in 'PTRATIO' will lead to a decrease in MEDV\n"





# TODO: Import 'r2_score'
from sklearn.metrics import r2_score


def performance_metric(actual, predicted):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(actual, predicted)

    # Return the score
    return score

# Question 2 - Goodness of Fit
print "Question 2 - Question 2 - Goodness of Fit:\n"
## Given a hypothetical model, determine if it successfully captures the variation of the target value based on the model's R2 score
score = performance_metric([3, 5, 2, 7, 4.2], [2.6, 4.3, 1.5, 6.4, 5])
print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)

## based on the model's R^2 score of 0.871, I would say the model successfully captured the variation of the target variables


## Training and testing
## First Data has to be randomized/ shuffled, then split into train-test blocks

# TODO: Import 'train_test_split'
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.3, shuffle=True)
print "3: Training and Testing"
print "Training and testing split was successful.\n"

print "Question: Reason why a dataset is split into training and testing subsets for a model."
print "Answer: to help with validation, precision and accuracy of your model. if entire dataset is used in testing, the dataset will not be generalized. The model will already know the answers and this might lead to problems of OVERFITTING"
print "Splitting into two subsets helps with optimization while leaving dtat for you to validate accuracy of the model"


# Question 3 - Analyzing Model Performance
print "Analyzing Model Performance:\n"
print "4. Learning Data"
print "Question: Student correctly identifies the trend of both the training and testing curves from the graph as more training points are added. Discussion is made as to whether additional training points would benefit the model."
#display.ModelLearning(features, prices)

print "Answer: As more training points are added, its score decreases and seems to level off, while its variance / uncertainty of the curve also decreasing. The testing curve's score also increases as more data points are provided up until approximately 300, as it then tends to level off and run parallel with the training curve.Given that both training and validation curves have levelled off, providing more training points would not benefit the model with significant improvements but may only increase time consumption for training and testing."


#display.ModelComplexity(features, prices)

print "5. Bias-Variance Tradeoff"
print "Question: Student correctly identifies whether the model at a max depth of 1 and a max depth of 10 suffer from either high bias or high variance, with justification using the complexity curves graph."
print "Answer: With a maximumn depth of 1, based on the complexity curve both the training and validation scores are low, the model suffers from high bias (underfitting). And at a maximum depth of 10 the model appears to suffer from high variance (overfitting)."
print "I can justify my conclusions based on the visual cues of the consistent variance on the validation score, in combination with the convergence of training and validation curves at max-depth of 1 and the large gap between the training and validation curves at a max-depth of 10."
print "\n"

print "6. Best-Guess Optimal Model"
print "Question: Which maximum depth do you think results in a model that best generalizes to unseen data? What intuition lead you to this answer?"
print "Answer: I believe a max-depth of 3 would result in the best generalized model. At a max-depth of 3, both validation and training curves are at their smallest/closest level of uncertainty between each other, while the validation score is near its highest value.\n"

print "7. Grid Search"
print "Question: What is the grid search technique and how it can be applied to optimize a learning algorithm?"
print "Answer: The grid search technique automates the process of tuning parameters of a model in order to get the best performance. For example, on a decision tree, you may want to find the best performance with max-depth (3 or 4) and criterion (entropy and gini). Grid search combines these parameter options for you, as in the table below, and allows for faster experiments to help optimize the learning."
print "\n"

print "8. Cross Validation"
print "Question: What is the k-fold cross-validation training technique? What benefit does this technique provide for grid search when optimizing a model?"
print "Answer: The k-fold cross-validation training technique is the process of dividing your data points into smaller number of k bins. Testing then occurs on one of the k bins while training occurs with the other k-1 bins. This process, testing and training, occurs k times across all bins for testing and training. The average of the k testing experiments are used as the overall result of the model."
print "Although grid search automates the parameter selection and tuning for best performance, not using cross-validation could result in the model being tuned only to a specific subset of data. This is because without using a technique such as cross-validation, for example, only using kfold to create testing and training data, will not shuffle your data points, i.e if your dataset is ordered or in any pattern, grid search would only perform tuning on the same subset of training data. Utilizing cross-validation, eliminates this issue by using the entire dataset allowing grid search to optimize parameter tuning across all data points."

print "Fitting a Model:\n"

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], test_size=0.20, random_state=0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': range(1, 11)}  # last value in range is exclusive

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

print "Model has been fit to parameters \n"

print "9. Optimal Model\n"
model =  fit_model(features, prices)
# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(model.get_params()['max_depth'])
print "\n"

print "10. Predicting Selling Prices"
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(model.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)

print "I would recommend each client to sell for the following with the given explanation."
print "client 1 - At 408k the house is within the mean of approx 454k. with 5 rooms, relatively average poverty level, also average student-teacher ratio. Its a good selling price"
print "client 2 - Minimum selling price is a bit above 100k, however the poverty rate and student teacher rate are quite lower than those of 100k to take the price up a bit higher"
print "client 3 - with a large number of rooms, very low neigborhood poverty levels and low student teacher ratio, the home price is near the maximum selling price in the neighborhood and is reasonably priced"

display.PredictTrials(features, prices, fit_model, client_data)

print "11. Applicability\n"
print "Question: In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting."
print "The constructed model, as is, should not be used in a real-world setting. There are a number of reasons for this answer and below I've highlighted a few:\n"

print "Relevancy: The data which the current model has been trained on, collected in 1978, is not relevant today (2016)"
print "Applicable: A model training on data from a city such as Boston, is not suitable to be used in urban areas such as Ohio nor would be it applicable for some other cities such as San Francisco"
print "Features: Although the dataset covers features which are present in today's homes, it is missing features that could affect the selling price in today's housing market such as size of a backyard or approximity to public transit if the home is in a large city."
print "Robustness: The current model appears to be too sensivity/not well generalized as running it multiple times for a specific client (as seen above) provides a wide variance in pricing, which as is would be unsatifactory in the real-world."
print "Supplying the model with more data, between 1978 and 2016, along with using a few additional features, the model may be robust and accurate enough to be applied to data from cities similar to Boston in the real-world."
