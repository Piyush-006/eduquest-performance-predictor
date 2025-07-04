import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🎓 Student Performance Predictor - EduQuest")

# Upload file
uploaded_file = st.file_uploader("Upload Student Dataset CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Grade mapping
    def label_grade(score):
        if score < 60:
            return 'At Risk'
        elif score < 80:
            return 'Average'
        else:
            return 'Excellent'

    df['grade_category'] = df['final_exam_score'].apply(label_grade)
    df.drop(columns=['final_exam_score'], inplace=True)

    # Encode categoricals
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Prepare data
    X = df.drop(columns=['grade_category'])
    y = df['grade_category']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y)

    # Show predictions
    y_pred = clf.predict(X_scaled)
    df['Predicted_Grade'] = y_pred

    st.success("✅ Prediction Complete")
    st.write(df[['Predicted_Grade']])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", data=csv, file_name="predicted_students.csv", mime='text/csv')
