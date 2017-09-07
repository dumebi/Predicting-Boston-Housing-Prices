Boston housing dataset has 489 data points with 4 variables each.

Statistics for Boston housing dataset:

Minimum price: $105,000.00 <br>
Maximum price: $1,024,800.00 <br>
Mean price: $454,342.13<br>
Median price $438,900.00<br>
Standard deviation of prices: $165,171.68


Question 1 - Feature Observation:

Increase in 'RM' will lead to an increase in MEDV<br>
Increase in 'LSTAT' will lead to a decrease in MEDV<br>
Increase in 'PTRATIO' will lead to a decrease in MEDV

Question 2 - Goodness of Fit:

Model has a coefficient of determination, R^2, of 0.871.

Question 3 - Training and Testing:

Training and testing split was successful.

Question: Reason why a dataset is split into training and testing subsets for a model.<br><br>
Answer: to help with validation, precision and accuracy of your model. if entire dataset is used in testing, the dataset will not be generalized.<br><br> The model will already know the answers and this might lead to problems of OVERFITTING
Splitting into two subsets helps with optimization while leaving dtat for you to validate accuracy of the model
Analyzing Model Performance:

Question 4 -  Learning Data:

Question: Student correctly identifies the trend of both the training and testing curves from the graph as more training points are added. Discussion is made as to whether additional training points would benefit the model.<br>
<br><br>Answer: As more training points are added, its score decreases and seems to level off, while its variance / uncertainty of the curve also decreasing. The testing curve's score also increases as more data points are provided up until approximately 300, as it then tends to level off and run parallel with the training curve.Given that both training and validation curves have levelled off, providing more training points would not benefit the model with significant improvements but may only increase time consumption for training and testing.

Question 5 - Bias-Variance Tradeoff:

Question: Student correctly identifies whether the model at a max depth of 1 and a max depth of 10 suffer from either high bias or high variance, with justification using the complexity curves graph.<br><br>
Answer: With a maximumn depth of 1, based on the complexity curve both the training and validation scores are low, the model suffers from high bias (underfitting). And at a maximum depth of 10 the model appears to suffer from high variance (overfitting).
<br><br>I can justify my conclusions based on the visual cues of the consistent variance on the validation score, in combination with the convergence of training and validation curves at max-depth of 1 and the large gap between the training and validation curves at a max-depth of 10.


Question 6 - Best-Guess Optimal Model:

Question: Which maximum depth do you think results in a model that best generalizes to unseen data? What intuition lead you to this answer?
<br><br>Answer: I believe a max-depth of 3 would result in the best generalized model. At a max-depth of 3, both validation and training curves are at their smallest/closest level of uncertainty between each other, while the validation score is near its highest value.

Question 7 - Grid Search:

Question: What is the grid search technique and how it can be applied to optimize a learning algorithm?
<br><br>Answer: The grid search technique automates the process of tuning parameters of a model in order to get the best performance. For example, on a decision tree, you may want to find the best performance with max-depth (3 or 4) and criterion (entropy and gini). Grid search combines these parameter options for you, as in the table below, and allows for faster experiments to help optimize the learning.


Question 8 - Cross Validation

Question: What is the k-fold cross-validation training technique? What benefit does this technique provide for grid search when optimizing a model?
<br><br>Answer: The k-fold cross-validation training technique is the process of dividing your data points into smaller number of k bins. Testing then occurs on one of the k bins while training occurs with the other k-1 bins. This process, testing and training, occurs k times across all bins for testing and training. The average of the k testing experiments are used as the overall result of the model.<br><br>
Although grid search automates the parameter selection and tuning for best performance, not using cross-validation could result in the model being tuned only to a specific subset of data. This is because without using a technique such as cross-validation, for example, only using kfold to create testing and training data, will not shuffle your data points, i.e if your dataset is ordered or in any pattern, grid search would only perform tuning on the same subset of training data. Utilizing cross-validation, eliminates this issue by using the entire dataset allowing grid search to optimize parameter tuning across all data points.


Fitting a Model:

Model has been fit to parameters 

Question 9 - Optimal Model:

Parameter 'max_depth' is 4 for the optimal model.


Question 10 - Predicting Selling Prices:

Predicted selling price for Client 1's home: $408,800.00<br>
Predicted selling price for Client 2's home: $231,253.45<br>
Predicted selling price for Client 3's home: $938,053.85<br>
I would recommend each client to sell for the following with the given explanation.<br>
<br>client 1 - At 408k the house is within the mean of approx 454k. with 5 rooms, relatively average poverty level, also average student-teacher ratio. Its a good selling price
<br><br>client 2 - Minimum selling price is a bit above 100k, however the poverty rate and student teacher rate are quite lower than those of 100k to take the price up a bit higher
<br><br>client 3 - with a large number of rooms, very low neigborhood poverty levels and low student teacher ratio, the home price is near the maximum selling price in the neighborhood and is reasonably priced
<br>Trial 1: $391,183.33<br>
Trial 2: $411,417.39<br>
Trial 3: $415,800.00<br>
Trial 4: $420,622.22<br>
Trial 5: $413,334.78<br>
Trial 6: $411,931.58<br>
Trial 7: $399,663.16<br>
Trial 8: $407,232.00<br>
Trial 9: $402,531.82<br>
Trial 10: $413,700.00<br>

Range in prices: $29,438.89

Question 11 - Applicability

Question: In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.
<br><br>The constructed model, as is, should not be used in a real-world setting. There are a number of reasons for this answer and below I've highlighted a few:

Relevancy: The data which the current model has been trained on, collected in 1978, is not relevant today (2016)
<br><br>Applicable: A model training on data from a city such as Boston, is not suitable to be used in urban areas such as Ohio nor would be it applicable for some other cities such as San Francisco
<br><br>Features: Although the dataset covers features which are present in today's homes, it is missing features that could affect the selling price in today's housing market such as size of a backyard or approximity to public transit if the home is in a large city.
<br><br>Robustness: The current model appears to be too sensivity/not well generalized as running it multiple times for a specific client (as seen above) provides a wide variance in pricing, which as is would be unsatifactory in the real-world.
<br><br>Supplying the model with more data, between 1978 and 2016, along with using a few additional features, the model may be robust and accurate enough to be applied to data from cities similar to Boston in the real-world.
