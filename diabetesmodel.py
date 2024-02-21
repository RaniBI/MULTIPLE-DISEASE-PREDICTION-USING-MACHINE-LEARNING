import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

# Load your dataset (replace 'diabetes_dataset.csv' with your dataset file)
data = pd.read_csv('diabetes_dataset.csv')

# Separate features (X) and target variable (y)
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Evaluate the Random Forest model
random_forest_accuracy = random_forest_model.score(X_test, y_test)
print(f"Random Forest Model Accuracy: {random_forest_accuracy:.2f}")

# Save the Random Forest model to a file
with open('random_forest_diabetes_model.sav', 'wb') as model_file:
    pickle.dump(random_forest_model, model_file)

# Train a Support Vector Machine (SVM) classifier
svm_model = SVC(kernel='linear', C=1, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate the SVM model
svm_accuracy = svm_model.score(X_test, y_test)
print(f"SVM Model Accuracy: {svm_accuracy:.2f}")

# Save the SVM model to a file
with open('svm_diabetes_model.sav', 'wb') as model_file:
    pickle.dump(svm_model, model_file)
