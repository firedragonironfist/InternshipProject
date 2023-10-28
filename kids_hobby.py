#!C:\Users\91636\AppData\Local\Programs\Python\Python311\python.exe

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv("Hobby_Data.csv")

# Label encode categorical columns
categorical_cols = ['Olympiad_Participation', 'School', 'Fav_sub', 'Projects', 'Medals', 'Career_sprt', 'Act_sprt', 'Fant_arts', 'Won_arts', 'Predicted Hobby']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define input and output columns
input_cols = ['Olympiad_Participation', 'School', 'Fav_sub', 'Projects', 'Grasp_pow', 'Time_sprt', 'Medals', 'Career_sprt', 'Act_sprt', 'Fant_arts', 'Won_arts', 'Time_art']
output_cols = ['Predicted Hobby']

X = df[input_cols]
Y = df[output_cols]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Create and train the SVM classifier
classifier = SVC(kernel='poly', C=10, gamma='scale')
classifier.fit(X_train, np.ravel(Y_train, order="c"))

# Print the model's score on the test set
score = classifier.score(X_test, Y_test)
print(f"Model Score on Test Set: {score}")

# Save the trained model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

print("Model saved to 'model.pkl'")
