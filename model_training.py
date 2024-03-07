# model_training.py
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import joblib
import pickle

# Load the preprocessed data
df = pd.read_csv('/home/ubuntu/Streamlit/Supply_Chain_preprocessed.csv')

# Separate df into positive, neutral and negative data
df_neg = df[df["sentiment"] == "Negative"]
df_pos = df[df["sentiment"] == "Positive"]
df_neutral = df[df["sentiment"] == "Neutral"]

# Joining the reviews into a single string for each sentiment
neg_reviews = ' '.join(df_neg["Text"].values.astype(str))
pos_reviews = ' '.join(df_pos["Text"].values.astype(str))
neutral_reviews = ' '.join(df_neutral["Text"].values.astype(str))

# Splitting into training and testing set
X, y = np.asarray(df.Text), np.asarray(df.sentiment)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Using CountVectorizer to transform
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

options = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'GradientBoosting']

for classifier in options:
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()
    elif classifier == 'Decision Tree':
        clf = DecisionTreeClassifier()
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()
    elif classifier == 'GradientBoosting':
        clf = GradientBoostingClassifier()

    # Train the classifier
    clf.fit(X_train, y_train)

    # Save the trained model using Joblib
    model_filename = f"{classifier.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(clf, model_filename)

    # Calculate and save accuracy
    accuracy = clf.score(X_test, y_test)
    accuracy_filename = f"{classifier.lower().replace(' ', '_')}_accuracy.pkl"
    with open(accuracy_filename, 'wb') as accuracy_file:
        pickle.dump(accuracy, accuracy_file)

    # Calculate and save confusion matrix
    confusion_matrix_result = confusion_matrix(y_test, clf.predict(X_test))
    confusion_matrix_filename = f"{classifier.lower().replace(' ', '_')}_confusion_matrix.pkl"
    with open(confusion_matrix_filename, 'wb') as confusion_matrix_file:
        pickle.dump(confusion_matrix_result, confusion_matrix_file)