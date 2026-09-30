import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading datasets...")

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["class"] = 0
true["class"] = 1

data = pd.concat([fake, true], axis=0)

data["text"] = data["text"].fillna("")

X = data["text"]
y = data["class"]

print("Total rows:", len(data))
print("Fake:", (y == 0).sum())
print("Real:", (y == 1).sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

print("Training TF-IDF...")

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("Features:", X_train.shape[1])
print("Training Logistic Regression...")

lr = LogisticRegression()
lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print()
print("========== RESULTS ==========")
print("Accuracy:", accuracy)
print()
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(lr, "lr_model_new.jb")
joblib.dump(vectorizer, "vectorizer_new.jb")

print()
print("New model saved:")
print("lr_model_new.jb")
print("vectorizer_new.jb")
