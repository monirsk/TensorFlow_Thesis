# TensorFlow_Thesis



Module 1:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Defining KNN

import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances(self, x):
        differences = (self.X_train - x)**2
        sum_squared = np.sum(differences, axis=1)     #distance from a single test point to each training sample, basically row wise jog korbe
        distances = np.sqrt(sum_squared)
        return distances


    def get_k_nearest_neighbors(self, distances):
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[:self.k]
        return k_indices

    def majority_vote(self, k_indices):
        k_nearest_labels = []
        for i in k_indices:
            label = self.y_train[i]
            k_nearest_labels.append(label)

        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1


        majority_label = None
        max_count = -1
        for label in label_counts:
            if label_counts[label] > max_count:
                max_count = label_counts[label]
                majority_label = label
        return majority_label



    def _predict_single_point(self, x):
        distances = self.compute_distances(x)
        k_indices = self.get_k_nearest_neighbors(distances)
        prediction = self.majority_vote(k_indices)

        return prediction

    def predict(self, X):
        predictions = []
        for x in X:
            label = self._predict_single_point(x)
            predictions.append(label)

        return np.array(predictions)



url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv(url, header=None, names=column_names)



X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNN(k=13)
knn.fit(X_train, y_train)

# Prediction on test data
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")




Module 1: Second Part


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Defining KNN

import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances(self, x):
        differences = (self.X_train - x)**2
        sum_squared = np.sum(differences, axis=1)     #distance from a single test point to each training sample, basically row wise jog korbe
        distances = np.sqrt(sum_squared)
        return distances


    def get_k_nearest_neighbors(self, distances):
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[:self.k]
        return k_indices

    def majority_vote(self, k_indices):
        k_nearest_labels = []
        for i in k_indices:
            label = self.y_train[i]
            k_nearest_labels.append(label)

        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1


        majority_label = None
        max_count = -1
        for label in label_counts:
            if label_counts[label] > max_count:
                max_count = label_counts[label]
                majority_label = label
        return majority_label



    def _predict_single_point(self, x):
        distances = self.compute_distances(x)
        k_indices = self.get_k_nearest_neighbors(distances)
        prediction = self.majority_vote(k_indices)

        return prediction

    def predict(self, X):
        predictions = []
        for x in X:
            label = self._predict_single_point(x)
            predictions.append(label)

        return np.array(predictions)



url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv(url, header=None, names=column_names)



X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values


import random
noise_ratio = 0.1  # 10% noise
y_noisy = y.copy()
n_samples = int(len(y) * noise_ratio)

indices = random.sample(range(len(y)), n_samples)
for idx in indices:
    # Flipping the binary labels
    y_noisy[idx] = 1 - y_noisy[idx]

# Split the noisy dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNN(k=13)

# Train the classifier on the noisy training data
knn.fit(X_train, y_train)

# Predict on the noisy test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for distorted dataset: {accuracy * 100:.2f}%")


