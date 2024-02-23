import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_wine

"""

Predicting Wine Quality with Random Forests. This project will guide you through the process of using a Random Forest
classifier to predict the quality of wines based on their chemical properties. The dataset we'll use is the Wine
Quality dataset, which contains information on various wines and a quality rating for each wine.

Project Overview Objective: Predict the quality of wine (classified into categories) based on its chemical properties
 using the Random Forest algorithm.

Dataset: The Wine Quality dataset, available in Scikit-learn or UCI Machine Learning Repository. It includes
parameters such as alcohol content, acidity, sugar level, and other chemical properties, along with a quality rating.

"""

# Load the wine dataset
data = load_wine()
X = data.data
y = data.target

'''Preparing the Data'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

'''Building and Training the Random Forest Model'''

# Initialize the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

'''Making Predictions and Evaluating the Model'''

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

'''
Explanation of Code
Load Dataset: We load the wine dataset from Scikit-learn, containing features in X and labels in y.

Splitting Data: 
We divide the dataset into training and testing sets, with 70% of the data used for training and 30% for testing.

Random Forest Model: 
We initialize a Random Forest classifier with 100 trees and train it on the training set.

Making Predictions: 
We use the trained model to predict the quality of wine in the test set.

Evaluating the Model: 
The performance of our model is evaluated using accuracy and a detailed classification report, 
which includes precision, recall, and F1-score for each class.

Understanding the Output Accuracy: 
This metric tells you the percentage of correctly predicted wine qualities in the test set. 

Classification Report: 
Provides a breakdown of precision, recall, and F1-score for each wine quality class. 
These metrics give you a deeper insight into how well the model performs for each class, especially in imbalanced 
datasets where some classes might have more samples than others.
'''