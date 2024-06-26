import numpy as np
import pandas as pd

target = 'churn'

# Get the categoricals
categoricals = ['gender', 'senior_citizen', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'internet_service', 'online_security',
                'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing', 'payment_method', 'churn']

# Categorical features
categorical_features = [
    column for column in categoricals if column not in target]

# Services columns
services = ['online_security', 'online_backup', 'device_protection',
            'tech_support', 'streaming_tv', 'streaming_movies']

# Infer values of missing total charges in the numerical columns through Function Transformer


def infer_missing_total_charge(df):
    # Creating a mask variable for the missing values in the column for totalcharges
    mask = df['total_charges'].isna()

    # Filling the missing values of total_charge with the values of the monthly_charge times tenure
    monthly_charges = df.loc[mask, 'monthly_charges']

    # If tenure is 0, times by 1 or tenure = 1
    tenure = df.loc[mask, 'tenure'].apply(lambda x: x+1 if x == 0 else x)

    df['total_charges'].fillna(monthly_charges*tenure, inplace=True)

    return df


def infer_missing_multiple_lines(df):
    mask = df['multiple_lines'].isna()

    # Get the values of the phone_service for missing multiple_lines
    phone_service = df.loc[mask, 'phone_service']

    # If phone_service is not available or No, then the value for multiple_lines is also No otherwise the value for multiple_lines remains missing
    multiple_lines = phone_service.apply(lambda x: x if x == 'No' else pd.NA)

    df['multiple_lines'].fillna(multiple_lines, inplace=True)

    return df


def feature_creation(X):
    # After imputation
    df_copy = pd.DataFrame(X, columns=categorical_features)

    # Create new feature in phone_service column- single or multiple lines, drop multiple_lines column
    # Create 'call_service' column if it doesn't exist
    if 'call_service' not in df_copy.columns:
        conditions = [
            (df_copy['multiple_lines'] == 'Yes') & (
                df_copy['phone_service'] == 'Yes'),
            (df_copy['multiple_lines'] == 'No') & (
                df_copy['phone_service'] == 'Yes')
        ]
        choices = ['Multiplelines', 'Singleline']
        df_copy['call_service'] = np.select(conditions, choices, default='No')

    # Create new feature from services column- security_service and streaming_service
    # Create 'security_service' column if it doesn't exist
    if 'security_service' not in df_copy.columns:
        conditions = [
            (df_copy['online_security'] == 'Yes') & (df_copy['online_backup'] == 'Yes') & (
                df_copy['device_protection'] == 'Yes') & (df_copy['tech_support'] == 'Yes'),
            (df_copy['online_security'] == 'Yes') & (df_copy['online_backup'] == 'Yes') & (
                df_copy['device_protection'] == 'No') & (df_copy['tech_support'] == 'No'),
            (df_copy['online_security'] == 'No') & (df_copy['online_backup'] == 'No') & (
                df_copy['device_protection'] == 'Yes') & (df_copy['tech_support'] == 'No'),
            (df_copy['online_security'] == 'No') & (df_copy['online_backup'] == 'No') & (
                df_copy['device_protection'] == 'No') & (df_copy['tech_support'] == 'Yes')
        ]
        choices = ['Fullsecurity', 'Securitybackup',
                   'Deviceprotection', 'Techsupport']
        df_copy['security_service'] = np.select(
            conditions, choices, default='No')

    # Create 'streaming_service' column if it doesn't exist
    if 'streaming_service' not in df_copy.columns:
        # streaming_service feature
        conditions = [
            (df_copy['streaming_tv'] == 'Yes') & (
                df_copy['streaming_movies'] == 'Yes'),  # Fullservice
            (df_copy['streaming_tv'] == 'Yes') & (
                df_copy['streaming_movies'] == 'No'),  # Tv
            (df_copy['streaming_tv'] == 'No') & (
                df_copy['streaming_movies'] == 'Yes')  # Movies
        ]
        choices = ['Fullservice', 'Tv', 'Movies']
        df_copy['streaming_service'] = np.select(
            conditions, choices, default='No')

    # Drop redundant feature columns- multiple_lines, services
    columns = ['phone_service', 'multiple_lines'] + services

    df_copy.drop(columns=columns, inplace=True, errors='ignore')

    return df_copy


def infer_missing_services(df):
    for service in services:
        mask = df[service].isna()

        # Get the values of the internet_service for missing service column
        internet_service = df.loc[mask, 'internet_service']

        # If internet_service is not available or No, then the value for multiple_lines is also No otherwise the value for multiple_lines remains missing
        fill_service = internet_service.apply(
            lambda x: x if x == 'No' else pd.NA)

        df[service].fillna(fill_service, inplace=True)

    return df
