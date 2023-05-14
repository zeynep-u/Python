import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

data = pd.read_csv("data_aug.csv")

feature_cols = ["age_upon_outcome", "sex_upon_outcome",	 "breed", "color"]

X = data[feature_cols] # Features
y = data.animal_type # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 70% training and 30% test
# # Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=10, min_samples_split=2)

# # Train Decision Tree Classifer
clf = clf.fit(X_train.iloc[:70000], y_train.iloc[:70000])

# #Predict the response for test dataset
y_pred = clf.predict(X_test)
# # Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
cart_params = {"max_depth": [1,3,5,8,10],
              "min_samples_split": [2,3,5,10,20,50]}
cart_cv_model = GridSearchCV(clf, cart_params, cv = 10, n_jobs = -1, verbose =2).fit(X_train, y_train)

print(cart_cv_model.best_params_)