from getsession import create_session_object, close_session

def create_table_in_snowflake(session, db, sch):
    """
    Create the CHURN table in Snowflake.
   
    Args:
        session (Session): The Snowflake session.
        db (str): The database name.
        sch (str): The schema name.
    """
    create_table_sql = f"""
    create or replace TABLE CHURN.PUBLIC.CHURN_DATA (
	CUSTOMERID VARCHAR(16777216),
	GENDER VARCHAR(16777216),
	SENIORCITIZEN NUMBER(38,0),
	PARTNER VARCHAR(16777216),
	DEPENDENTS VARCHAR(16777216),
	TENURE NUMBER(38,0),
	PHONESERVICE VARCHAR(16777216),
	MULTIPLELINES VARCHAR(16777216),
	INTERNETSERVICE VARCHAR(16777216),
	ONLINESECURITY VARCHAR(16777216),
	ONLINEBACKUP VARCHAR(16777216),
	DEVICEPROTECTION VARCHAR(16777216),
	TECHSUPPORT VARCHAR(16777216),
	STREAMINGTV VARCHAR(16777216),
	STREAMINGMOVIES VARCHAR(16777216),
	CONTRACT VARCHAR(16777216),
	PAPERLESSBILLING VARCHAR(16777216),
	PAYMENTMETHOD VARCHAR(16777216),
	MONTHLYCHARGES FLOAT,
	TOTALCHARGES FLOAT,
	CHURN VARCHAR(16777216)
    )
    """
    try:
        session.sql(create_table_sql).collect()
        print(f"Table '{db}.{sch}.CHURN_DATA' created successfully.")
    except Exception as e:
        print(f"Error creating table '{db}.{sch}.house_price':", e)

def load_csv_into_snowflake(session, db, sch, csv_file_path):
    """
    Load data from a CSV file into the CHURN table in Snowflake.
   
    Args:
        session (Session): The Snowflake session.
        db (str): The database name.
        sch (str): The schema name.
        csv_file_path (str): The path to the CSV file.
    """
    try:
        # Escape backslashes in the file path
        escaped_csv_file_path = csv_file_path.replace("\\", "/")
        
        # Stage the CSV file with the corrected path
        stage_file_sql = f"PUT 'file:///{escaped_csv_file_path}' @STG"
        session.sql(stage_file_sql).collect()
        print(f"File '{csv_file_path}' staged successfully.")
        
        # Extract the file name from the path
        file_name = csv_file_path.split("\\")[-1]
        
        # Load data into the table, specifying that the file contains a header row
        copy_into_sql = f"""
        COPY INTO {db}.{sch}.CHURN_DATA
        FROM @STG/{file_name}
        FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"', SKIP_HEADER = 1)
        ON_ERROR = 'CONTINUE';
        """
        session.sql(copy_into_sql).collect()
        print(f"Data loaded into table '{db}.{sch}.CHURN' successfully.")
    except Exception as e:
        print(f"Error loading data into table '{db}.{sch}.CHURN':", e)

if __name__ == "__main__":
    # Step 1: Create a Snowflake session
    session = create_session_object()



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

    if session:
        # Step 2: Define database, schema, and CSV file path
        db = "CHURN"
        sch = "public"
        csv_file_path = "./datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"

        # Step 3: Create the table in Snowflake
        create_table_in_snowflake(session, db, sch)

        # Step 4: Load CSV data into the table
        load_csv_into_snowflake(session, db, sch, csv_file_path)

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
