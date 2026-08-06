"""
train_models.py
---------------
Trains the complaint classifier and saves model artifacts to the models/ directory.
Run this once before launching the Streamlit app:
    python train_models.py
"""

import os
import re
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Download NLTK data ---
print("[1/6] Downloading NLTK data...")
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(pkg, quiet=True)
    print(f"  {pkg}: ready")

# --- 2. Load dataset ---
print("\n[2/6] Loading dataset...")
DATA_PATH = os.path.join('data', 'complaints.csv')
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'. "
                            "Please run data/generate_dataset.py first.")

df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df)} records with columns: {list(df.columns)}")

# --- 3. Pre-process text ---
print("\n[3/6] Pre-processing text...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['complaint_text'].apply(clean_text)
print(f"  Done. Sample: {df['cleaned_text'].iloc[0][:80]}...")

# --- 4. Encode labels and vectorize ---
print("\n[4/6] Encoding labels & vectorizing...")
le = LabelEncoder()
y = le.fit_transform(df['category'])
print(f"  Classes: {list(le.classes_)}")

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text'])
print(f"  Feature matrix shape: {X.shape}")

# --- 5. Train model ---
print("\n[5/6] Training Logistic Regression model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"  Test Accuracy: {acc * 100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- 6. Save artifacts ---
print("[6/6] Saving model artifacts...")
os.makedirs('models', exist_ok=True)
joblib.dump(model,  os.path.join('models', 'complaint_classifier.pkl'))
joblib.dump(tfidf,  os.path.join('models', 'tfidf_vectorizer.pkl'))
joblib.dump(le,     os.path.join('models', 'label_encoder.pkl'))

for fname in ['complaint_classifier.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl']:
    path = os.path.join('models', fname)
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✓ {fname}  ({size_kb:.1f} KB)")

print("\n✅ All model artifacts saved successfully to models/")
print("   You can now run the Streamlit app:  streamlit run app.py")
