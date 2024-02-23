# import numpy as np
# import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

'''
numpy is for handling arrays.
matplotlib.pyplot is for plotting graphs.
datasets from sklearn provides access to several datasets, including the Iris dataset.
train_test_split will help us split our data into training and testing sets.
StandardScaler is used for feature scaling.
KNeighborsClassifier is the K-Nearest Neighbors (KNN) classifier we'll use for this example.
classification_report and confusion_matrix are for evaluating the performance of our model.
'''

'''Load the Iris dataset and explore its structure.'''

# Load dataset
iris = datasets.load_iris()

# Explore the dataset structure
print(iris.keys())
print(iris['DESCR'])  # Uncomment to view a description of the dataset

'''Split the data into features (X) and labels (y), and then split these into training and test sets.'''

X = iris.data
y = iris.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''Before making any actual predictions, 
it's important to scale the features so that all of them can be uniformly evaluated.'''

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''We'll use the K-Nearest Neighbors (KNN) algorithm for our classification model.'''

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the data
knn.fit(X_train, y_train)

'''Use the trained model to make predictions on the test set.'''
y_pred = knn.predict(X_test)

'''After making predictions, it's important to see how well our model is performing.'''
# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

# Print classification report
print(classification_report(y_test, y_pred))

'''The confusion matrix gives you insight into the number of correct and incorrect predictions for each class, 
while the classification report provides key metrics like precision, recall, and f1-score for each class.'''

'''
    Experimenting and Improving Experiment with different numbers of neighbors (n_neighbors) in the KNN 
    classifier to see how it affects model accuracy. Additionally, explore other classifiers available in Scikit-learn 
    and compare their performance.
    
    Understanding the Process
    Step 2 sets up our toolset.
    Step 3 lets us peek into the data we're working with.
    Step 4 is about organizing our data into a format suitable for training/testing.
    Step 5 ensures our model doesn't get biased by the scale of features.
    Step 6 is where the learning happens.
    Step 7 applies our learned model to make predictions.
    Step 8 assesses how well our model is doing.
'''
