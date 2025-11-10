import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Page config
st.set_page_config(page_title="Sleep Disorder Prediction", page_icon="😴", layout="wide")

# Load dataset
st.title("😴 Sleep Disorder Prediction System")
df = pd.read_csv("data.csv")

# Data preprocessing
@st.cache_data
def preprocess_data(df):
    """Preprocess the dataset for training"""
    df_processed = df.copy()
    
    # Drop Person ID (not a feature)
    df_processed = df_processed.drop('Person ID', axis=1)
    
    # Split Blood Pressure into Systolic and Diastolic
    df_processed[['Systolic BP', 'Diastolic BP']] = df_processed['Blood Pressure'].str.split('/', expand=True)
    df_processed['Systolic BP'] = df_processed['Systolic BP'].astype(int)
    df_processed['Diastolic BP'] = df_processed['Diastolic BP'].astype(int)
    df_processed = df_processed.drop('Blood Pressure', axis=1)
    
    # Handle missing Sleep Disorder (NaN means no disorder - healthy)
    df_processed['Sleep Disorder'] = df_processed['Sleep Disorder'].fillna('None')
    
    # Store label encoders for later use
    label_encoders = {}
    
    # Encode categorical columns
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    return df_processed, label_encoders

# Preprocess data
df_processed, label_encoders = preprocess_data(df)

# Display dataset info
with st.expander("📊 View Dataset Information"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", df_processed.shape[1] - 1)
    with col3:
        disorder_counts = df['Sleep Disorder'].fillna('None').value_counts()
        st.metric("Healthy People", disorder_counts.get('None', 0))
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

# Train model
@st.cache_resource
def train_model(df_processed):
    """Train the Random Forest model"""
    X = df_processed.drop("Sleep Disorder", axis=1)
    y = df_processed["Sleep Disorder"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X.columns.tolist()

model, accuracy, feature_names = train_model(df_processed)

st.success(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")

# Prediction Section
st.markdown("---")
st.subheader("🔮 Make a Prediction")
st.write("Enter the patient's information below:")

# Create input form with a submit button
with st.form(key='prediction_form'):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Personal Info**")
        gender = st.selectbox("Gender", options=df['Gender'].unique())
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Occupation", options=sorted(df['Occupation'].unique()))
    
    with col2:
        st.markdown("**Sleep Metrics**")
        sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, 0.1)
        quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
        
        st.markdown("**Lifestyle**")
        physical_activity = st.slider("Physical Activity (min/day)", 0, 120, 60)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    
    with col3:
        st.markdown("**Health Metrics**")
        bmi_category = st.selectbox("BMI Category", options=sorted(df['BMI Category'].unique()))
        heart_rate = st.slider("Heart Rate (bpm)", 50, 120, 70)
        daily_steps = st.number_input("Daily Steps", 1000, 20000, 7000, step=100)
        
        systolic_bp = st.slider("Systolic BP", 90, 180, 120)
        diastolic_bp = st.slider("Diastolic BP", 60, 120, 80)
    
    # Submit button
    predict_button = st.form_submit_button(label="🔮 Predict Sleep Disorder", use_container_width=True)

    # Prediction logic (FIXED VERSION)
if predict_button:
    # Create input dictionary
    user_input = {
        'Gender': gender,
        'Occupation': occupation,
        'BMI Category': bmi_category,
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'Systolic BP': systolic_bp,
        'Diastolic BP': diastolic_bp
    }
    
    # Create DataFrame from user input
    user_df = pd.DataFrame([user_input])
    
    # CRITICAL: Use the STORED label encoders, NOT fit_transform!
    # This line must use label_encoders[col].transform() NOT le.fit_transform()
    for col in ['Gender', 'Occupation', 'BMI Category']:
        if col in label_encoders:
            user_df[col] = label_encoders[col].transform(user_df[col])
    
    # Debug: Print the encoded values
    st.write("**Debug - Encoded Input:**")
    st.dataframe(user_df)
    
    # Ensure columns are in the same order as training
    user_df = user_df[feature_names]
    
    # Make prediction
    try:
        prediction_encoded = model.predict(user_df)[0]
        prediction_proba = model.predict_proba(user_df)[0]
        
        # Decode prediction
        predicted_disorder = label_encoders['Sleep Disorder'].inverse_transform([prediction_encoded])[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📋 Prediction Results")
        
        # Main prediction
        if predicted_disorder == "None":
            st.success(f"### ✅ {predicted_disorder}")
            st.info("🎉 Great news! No sleep disorder detected. Keep maintaining your healthy lifestyle!")
        elif predicted_disorder == "Insomnia":
            st.warning(f"### ⚠️ {predicted_disorder}")
            st.warning("You may be experiencing insomnia. Consider consulting a healthcare professional.")
        else:  # Sleep Apnea
            st.error(f"### 🚨 {predicted_disorder}")
            st.error("Sleep Apnea detected. Please consult a sleep specialist immediately.")
        
        # Show probabilities
        st.subheader("📊 Prediction Confidence")
        disorder_classes = label_encoders['Sleep Disorder'].classes_
        
        prob_df = pd.DataFrame({
            'Condition': disorder_classes,
            'Probability': [f"{p*100:.2f}%" for p in prediction_proba],
            'Raw Probability': prediction_proba
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        # Debug info
        st.write("**Debug - Prediction Details:**")
        st.write(f"Encoded prediction: {prediction_encoded}")
        st.write(f"Disorder classes: {disorder_classes}")
        st.write(f"Feature names: {feature_names}")
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

        
        # Show input summary
        with st.expander("📝 View Input Summary"):
            input_display = user_input.copy()
            st.json(input_display)
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.code(f"Error details: {e}")

# Footer
st.markdown("---")
st.caption("⚕️ This is a predictive model for educational purposes. Always consult healthcare professionals for medical advice.")
