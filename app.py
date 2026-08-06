import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_text import LimeTextExplainer

# Download NLTK data if not already present
for _pkg in ['stopwords', 'punkt', 'punkt_tab', 'wordnet']:
    nltk.download(_pkg, quiet=True)

# --- Page Config ---
st.set_page_config(
    page_title="Customer Complaint Classifier",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .badge-high {
        background-color: #ff4b4b;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #ffa421;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #21c354;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    .badge-category {
        background-color: #1e88e5;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def get_resolution_priority(category, confidence):
    category_priority = {
        'Billing': 3,
        'Technical Support': 4,
        'Service Quality': 2,
        'Delivery': 3,
        'Account': 5
    }
    base_score = category_priority.get(category, 3)
    priority_score = base_score * confidence
    if priority_score >= 3.5:
        return 'High', priority_score
    elif priority_score >= 2.0:
        return 'Medium', priority_score
    else:
        return 'Low', priority_score

@st.cache_resource
def load_models():
    models_dir = 'models'
    try:
        classifier = joblib.load(os.path.join(models_dir, 'complaint_classifier.pkl'))
        vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
        encoder    = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        return classifier, vectorizer, encoder
    except FileNotFoundError:
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_data():
    data_path = 'data/complaints.csv'
    try:
        df = pd.read_csv(data_path)
        if 'date_received' in df.columns:
            df['date_received'] = pd.to_datetime(df['date_received'])
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# --- Main App ---
classifier, vectorizer, encoder = load_models()
df = load_data()

# --- Sidebar ---
st.sidebar.title("📊 Complaint Classifier")
st.sidebar.markdown("""
Welcome to the Customer Complaint Classification app. 
This academic project aims to classify customer complaints, analyze issue trends, and explain model predictions.
""")

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to:", 
                        ['Complaint Classifier', 'Issue Trends & Insights', 'Explainable AI'])

# Pipeline for LIME
def make_prediction_pipeline(texts):
    processed = [preprocess_text(t) for t in texts]
    features = vectorizer.transform(processed)
    return classifier.predict_proba(features)

# Check if models are loaded
if classifier is None or vectorizer is None or encoder is None:
    st.error("Error: Models not found. Please ensure that you have run the model training notebook and saved the models in the `models/` directory.")
    st.stop()

# --- Page Routing ---
if page == 'Complaint Classifier':
    st.title("Complaint Classifier")
    st.markdown("Enter a customer complaint below to predict its category, confidence score, and suggested resolution priority.")
    
    user_input = st.text_area("Customer Complaint:", height=150, placeholder="E.g., I have been overcharged on my last month's bill...")
    
    if st.button("Classify Complaint"):
        if not user_input.strip():
            st.warning("Please enter some text to classify.")
        elif len(user_input.split()) < 3:
            st.warning("The text is too short. Please enter a complete complaint.")
        else:
            with st.spinner("Classifying..."):
                # Preprocess and Predict
                processed_text = preprocess_text(user_input)
                features = vectorizer.transform([processed_text])
                prediction_idx = classifier.predict(features)[0]
                probabilities = classifier.predict_proba(features)[0]
                
                predicted_category = encoder.inverse_transform([prediction_idx])[0]
                confidence = probabilities[prediction_idx]
                
                # Priority
                priority_level, priority_score = get_resolution_priority(predicted_category, confidence)
                
                # Display Results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Predicted Category")
                    st.markdown(f"<span class='badge-category'>{predicted_category}</span>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Confidence Score")
                    st.metric(label="", value=f"{confidence*100:.1f}%")
                    st.progress(float(confidence))
                
                with col3:
                    st.markdown("### Suggested Priority")
                    badge_class = f"badge-{priority_level.lower()}"
                    st.markdown(f"<span class='{badge_class}'>{priority_level}</span>", unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### LIME Explanation")
                try:
                    explainer = LimeTextExplainer(class_names=encoder.classes_)
                    exp = explainer.explain_instance(user_input, make_prediction_pipeline, num_features=6, top_labels=1)
                    
                    # Display LIME explanation as HTML
                    html = exp.as_html()
                    st.components.v1.html(html, height=400, scrolling=True)
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")

elif page == 'Issue Trends & Insights':
    st.title("Issue Trends & Insights")
    if df is None:
        st.warning("Dataset not found at `data/complaints.csv`. Cannot display trends.")
    else:
        st.markdown("Explore the historical trends and distributions of customer complaints.")
        
        # Display Stats
        st.markdown("### Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Complaints", len(df))
        col2.metric("Categories", df['category'].nunique() if 'category' in df.columns else "N/A")
        
        if 'status' in df.columns:
            resolved = len(df[df['status'].str.lower() == 'resolved'])
            col3.metric("Resolved Complaints", f"{resolved} ({resolved/len(df)*100:.1f}%)")
        
        # Charts
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Category Distribution")
            if 'category' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(data=df, y='category', order=df['category'].value_counts().index, palette='viridis', ax=ax)
                ax.set_xlabel("Count")
                ax.set_ylabel("Category")
                st.pyplot(fig)
            else:
                st.info("Category column not found.")
                
        with col2:
            st.markdown("### Priority Distribution")
            if 'priority' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                priority_counts = df['priority'].value_counts()
                ax.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
                st.pyplot(fig)
            else:
                st.info("Priority column not found.")
                
        st.markdown("---")
        st.markdown("### Complaints Over Time")
        if 'date_received' in df.columns:
            time_df = df.groupby(df['date_received'].dt.to_period('M')).size().reset_index(name='count')
            time_df['date_received'] = time_df['date_received'].dt.to_timestamp()
            
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.lineplot(data=time_df, x='date_received', y='count', marker='o', ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Complaints")
            st.pyplot(fig)
        else:
            st.info("Date received column not found.")

elif page == 'Explainable AI':
    st.title("Explainable AI (XAI)")
    st.markdown("Understand how the model makes decisions using LIME (Local Interpretable Model-agnostic Explanations).")
    
    input_method = st.radio("Choose Input Method:", ["Select Sample from Dataset", "Custom Input"])
    
    text_to_explain = ""
    
    if input_method == "Select Sample from Dataset":
        if df is not None and 'complaint_text' in df.columns:
            sample_idx = st.selectbox("Select a sample:", df.index, format_func=lambda x: f"Complaint {df.loc[x, 'complaint_id']} - {df.loc[x, 'category']}" if 'complaint_id' in df.columns and 'category' in df.columns else f"Sample {x}")
            text_to_explain = df.loc[sample_idx, 'complaint_text']
            st.text_area("Selected Text:", text_to_explain, height=100, disabled=True)
            
            if 'category' in df.columns:
                st.write(f"**Actual Category:** {df.loc[sample_idx, 'category']}")
        else:
            st.warning("Dataset not available to select samples.")
    else:
        text_to_explain = st.text_area("Enter custom text to explain:", height=100)
        
    if st.button("Generate Explanation"):
        if not text_to_explain.strip():
            st.warning("Please provide text to explain.")
        else:
            with st.spinner("Generating LIME Explanation..."):
                try:
                    # Model prediction
                    processed_text = preprocess_text(text_to_explain)
                    features = vectorizer.transform([processed_text])
                    prediction_idx = classifier.predict(features)[0]
                    predicted_category = encoder.inverse_transform([prediction_idx])[0]
                    
                    st.markdown(f"### Predicted Category: **{predicted_category}**")
                    
                    # LIME
                    explainer = LimeTextExplainer(class_names=encoder.classes_)
                    exp = explainer.explain_instance(text_to_explain, make_prediction_pipeline, num_features=10, top_labels=1)
                    
                    # Display LIME explanation as HTML
                    st.markdown("### LIME Text Explanation")
                    html = exp.as_html()
                    st.components.v1.html(html, height=400, scrolling=True)
                    
                    # Feature Importance Chart
                    st.markdown("### Feature Importance (Top Words)")
                    exp_list = exp.as_list(label=prediction_idx)
                    words = [x[0] for x in exp_list]
                    weights = [x[1] for x in exp_list]
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ['#21c354' if w > 0 else '#ff4b4b' for w in weights]
                    ax.barh(words, weights, color=colors)
                    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
                    ax.set_xlabel("Weight")
                    ax.set_title(f"Features contributing to '{predicted_category}'")
                    ax.invert_yaxis()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")
