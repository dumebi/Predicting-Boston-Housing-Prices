import numpy as np
import pandas as pd

#Reading in and parsing data
raw_data = open('SMSSpamCollection.txt', 'r')
sms_data = []
for line in raw_data:
    split_line = line.split("\t")
    sms_data.append(split_line)

#Splitting data into messages and labels and training and test
sms_data = np.array(sms_data)
X = sms_data[:, 1]
y = sms_data[:, 0]

#Build a LinearSVC model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

#Build tf-idf vector representation of data
vectorizer = TfidfVectorizer()
vectorized_text = vectorizer.fit_transform(X)
text_clf = LinearSVC()
text_clf = text_clf.fit(vectorized_text, y)
#Test the model
print text_clf.predict(vectorizer.transform(["""SmartAlumni: Your Smart Alumni activation code is 17908 """]))
#Cross - Validation
from sklearn.model_selection import cross_val_score
cross_score = cross_val_score(text_clf, vectorized_text, y, cv=10)
print cross_score
print "mean:",np.mean(cross_score)