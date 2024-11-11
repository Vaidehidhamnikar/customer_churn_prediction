# Customer Churn Prediction App
This project is a machine learning-based web application that predicts customer churn for a telecommunications company. Built with Streamlit, it allows users to upload customer data, preprocess it, train a predictive model, and evaluate its performance on various metrics.

# Project Overview
Customer churn prediction helps identify customers who are likely to stop using a service. This app uses historical customer data to train a machine learning model that predicts churn probability. The model helps businesses take proactive measures to retain customers.

# Features
File Upload: Upload customer data in CSV format for processing and prediction.
Data Preprocessing: Cleans and encodes data, handling null values and categorical features.
Model Training: Trains a Random Forest classifier on customer data.
Evaluation Metrics: Displays accuracy, precision, recall, and F1-score to assess model performance.
Model Saving and Deployment: Saves the trained model and supports versioning via Snowflake for model deployment.

# Dataset
This app uses a sample dataset containing customer information, such as:

Demographics: Age, gender, seniority.
Service Usage: Tenure, internet service, contract type, etc.
Billing Details: Monthly charges, total charges, payment methods.
Target Variable: Churn (whether a customer has left the service or not).

The dataset used for training is located at: datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv
