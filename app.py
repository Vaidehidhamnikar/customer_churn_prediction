import streamlit as st
import pandas as pd
import numpy as np
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from sklearn.ensemble import RandomForestClassifier
# from snowflake.ml.modeling.ensemble import RandomForestClassifier
from snowflake.ml.registry import Registry
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from snowflake.ml.modeling.preprocessing import OneHotEncoder
from snowflake.ml.modeling.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from getsession import create_session_object, close_session


session = create_session_object()
if session is None:
    st.error("Failed to create Snowflake session. Please check your configuration.")
else:
    st.success("Successfully connected to Snowflake.")


st.title("Churn Prediction App")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(" Uploaded Data", df.head())

    st.header("Data Preprocessing")

    df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)
    df['TotalCharges'] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    df['tenure_bin'] = pd.cut(df['tenure'], bins=[-1, 24, 48, 72], labels=['New', 'Mid', 'Long'])

    columns_to_update = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df[columns_to_update] = df[columns_to_update].replace('No internet service', 'No')
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

    label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines', 'SeniorCitizen']
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    one_hot_cols = ['Contract', 'PaymentMethod', 'InternetService', 'tenure_bin']
    df = pd.get_dummies(df, columns=one_hot_cols)

    st.write(" Preprocessed Data", df.head())

    # st.header("Feature Selection")

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    X_numeric = X[numeric_cols]
    X_scaled = StandardScaler().fit_transform(X_numeric)

    st.header("Model Training and Evaluation")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    # st.write(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"### Model Evaluation")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    model_file_path = r'C:\Users\Vaidehi Dhamnikar\Downloads\model1.pkl'
    with open(model_file_path, "wb") as model_file:
        pickle.dump(rf_classifier, model_file)
    st.write(f"Model saved at {model_file_path}")


    db = 'CHURN' 
    schema = 'public' 
    model_name = 'CHURN_PREDICTION'

    native_registry = Registry(session=session, database_name=db, schema_name=schema)

    # model_ver = native_registry.log_model(
    #     model_name=model_name,
    #     version_name='V6',
    #     model= rf_classifier,
    #     sample_input_data=X_train,
    # )

    # model_ver.set_metric(metric_name="Accuracy is", value=accuracy)

    # model_ver.comment = " Churn Prediction based on features."

    model_versions = native_registry.get_model(model_name)
    
    st.write(model_versions.show_versions())

    close_session(session)
