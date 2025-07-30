import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("data/emails.csv")
df["text"] = df["domain"] + " " + df["subject"] + " " + df["body"]

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000)),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/email_auth_model.pkl")
print("Model trained and saved to models/email_auth_model.pkl")
