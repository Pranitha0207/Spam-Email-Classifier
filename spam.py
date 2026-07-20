import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\intern proj 2\Spam Email Classifier\spam.csv", encoding='latin-1')

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

# Split data
X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert text to vectors
vectorizer = CountVectorizer()

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()

model.fit(X_train_vector, y_train)

# Predictions
y_pred = model.predict(X_test_vector)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

# Test custom email
sample_email = [
    "spam,Your B4U voucher w/c 27/03 is MARSMS. Log onto www.B4Utele.com for discount credit. To opt out reply stop. Customer care call 08717168528,,,"
]

sample_vector = vectorizer.transform(sample_email)

prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam")
