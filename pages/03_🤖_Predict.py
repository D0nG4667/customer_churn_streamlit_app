import os
import random
import string
import time

import streamlit as st
import extra_streamlit_components as stx
import pandas as pd

import joblib

import datetime as dt

from config.settings import ENCODER_FILE, MODELS, HISTORY_FILE, TEST_FILE, TEST_FILE_URL
from includes.func_in_pipeline import *
from includes.logo import logo
from includes.janitor import Janitor
from includes.markdown import markdown_table_all


# Set page configuration
st.set_page_config(
    page_title='Predict Page',
    page_icon='🤖',
    layout="wide",
    initial_sidebar_state='auto'
)

# Use app logo
logo()


# All model paths
@st.cache_data(show_spinner=False)
def get_model_paths(MODELS):
    # List all pipelines in the models directory
    all_model_paths = [f for f in os.listdir(
        MODELS) if os.path.isfile(os.path.join(MODELS, f))]
    return all_model_paths


@st.cache_data(show_spinner=False)
def get_pipeline_names(model_paths):
    # All pipeline names
    all_pipeline_names = [p.split(".")[0] for p in model_paths]
    return all_pipeline_names


# Load pipelines
# For memory efficiency in loading the models, return a generator.
# Caching does not work well with generators
@st.cache_resource(show_spinner=False)
def load_pipelines(model_paths):
    def load_pipeline(model_path):
        return joblib.load(os.path.join(MODELS, model_path))

    # Generate all models
    pipelines = (load_pipeline(m)
                 for m in model_paths)  # Generator expression

    # Returning generators is buggy especially when multiple tabs are open, so return list of pipelines
    pipelines = [p for p in pipelines]

    return pipelines


# Set the selected pipeline key and best_model name
s_p_key = 'selected_model'
best_model_name = 'LogisticRegression'


@st.cache_resource(show_spinner=f"Loading{st.session_state.get(s_p_key, "")+" "}pipeline...")
def load_selected_pipeline(selected_pipeline_name, pipeline_names, model_paths):
    pipelines = load_pipelines(model_paths)

    index = pipeline_names.index(selected_pipeline_name)

    for i, pipeline in enumerate(pipelines):
        if i == index:
            selected_pipeline = pipeline
            break

    return selected_pipeline


# Load encoder
@st.cache_resource(show_spinner="Loading encoder...")
def load_encoder():
    return joblib.load(ENCODER_FILE)


@st.cache_data(show_spinner="Getting customer database..." if st.session_state.get('search_customer', False) else False)
def get_test_data():
    try:
        df_test_raw = pd.read_excel(TEST_FILE_URL)
    except Exception:
        df_test_raw = pd.read_excel(TEST_FILE)

    # Some house keeping, clean df
    df_test = df_test_raw.copy()
    janitor = Janitor()
    df_test = janitor.clean_dataframe(df_test)  # Cleaned

    return df_test_raw, df_test


def select_model():
    model_paths = get_model_paths(MODELS)
    pipeline_names = get_pipeline_names(model_paths)
    best_model_index = pipeline_names.index(best_model_name)

    col1, col2 = st.columns(2)

    with col1:
        selected_pipeline_name = st.selectbox('Select a model', options=pipeline_names,
                                              index=best_model_index, key=s_p_key)
    with col2:
        pass

    pipeline = load_selected_pipeline(
        selected_pipeline_name, pipeline_names, model_paths)

    encoder = load_encoder()

    return pipeline, encoder


def make_prediction(pipeline, encoder):
    _, df_test = get_test_data()
    df = None
    search_customer = st.session_state.get('search_customer', False)
    search_customer_id = st.session_state.get('search_customer_id', False)
    manual_customer_id = st.session_state.get('manual_customer_id', False)
    if isinstance(search_customer_id, str) and search_customer_id:  # And not empty string
        search_customer_id = [search_customer_id]
    if search_customer and search_customer_id:  # Search Form df and a customer was selected
        mask = df_test['customer_id'].isin(search_customer_id)
        df_form = df_test[mask]
        df = df_form.copy()
    elif not (search_customer or search_customer_id) and manual_customer_id:  # Manual form df
        columns = [
            'manual_customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
            'tenure', 'phone_service', 'multiple_lines', 'internet_service',
            'online_security', 'online_backup', 'device_protection', 'tech_support',
            'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing',
            'payment_method', 'monthly_charges', 'total_charges'
        ]
        data = {c: [st.session_state.get(c)] for c in columns}
        data['tenure'] = [int(d) for d in data['tenure']]
        data['monthly_charges'] = [float(d) for d in data['monthly_charges']]
        data['total_charges'] = [float(d) for d in data['total_charges']]

        # Make a DataFrame
        df = pd.DataFrame(data).rename(
            columns={'manual_customer_id': 'customer_id'})
    else:  # Form did not send a customer
        message = 'You must choose valid customer(s) from the select box.'
        icon = '😞'
        st.toast(message, icon=icon)
        st.warning(message, icon=icon)

    if df is not None:
        # Define Probability and Prediction
        pred = pipeline.predict(df)
        pred_int = int(pred[0])
        prediction = encoder.inverse_transform([pred_int])[0]
        probability = pipeline.predict_proba(df)[0][pred_int]*100

        # Store results in session state
        st.session_state['prediction'] = prediction
        st.session_state['probability'] = probability

        df['prediction'] = prediction
        df['probability (%)'] = probability
        df['time_of_prediction'] = dt.datetime.now()
        df['model_used'] = st.session_state['selected_model']

        df.to_csv(HISTORY_FILE, mode='a',
                  header=not os.path.isfile(HISTORY_FILE))

    return df


def convert_string(df: pd.DataFrame, string: str):
    return string.title().replace('_', '') if all('_' not in col for col in df.columns) else string


def make_predictions(pipeline, encoder, df_uploaded=None, df_uploaded_clean=None):
    df = None
    search_customer = st.session_state.get('search_customer', False)
    customer_id_bulk = st.session_state.get('customer_id_bulk', False)
    upload_bulk_predict = st.session_state.get('upload_bulk_predict', False)
    if search_customer and customer_id_bulk:  # Search Form df and a customer was selected
        _, df_test = get_test_data()
        mask = df_test['customer_id'].isin(customer_id_bulk)
        df_bulk = df_test[mask]
        df = df_bulk.copy()

    elif not (search_customer or customer_id_bulk) and upload_bulk_predict:  # Upload widget df
        df = df_uploaded_clean.copy()
    else:  # Form did not send a customer
        message = 'You must choose valid customer(s) from the select box.'
        icon = '😞'
        st.toast(message, icon=icon)
        st.warning(message, icon=icon)

    if df is not None:  # df should be set by form input or upload widget
        # Do predictions and probabilities
        preds = pipeline.predict(df)
        predictions = encoder.inverse_transform(preds)

        probabilities = pipeline.predict_proba(df)*100
        probabilities_preds = pd.DataFrame(probabilities)
        probabilities_preds[2] = preds

        def select_probability(row):
            return round(row[0], 2) if int(row[2]) == 0 else round(row[1], 2)
        probabilities = probabilities_preds.apply(
            lambda row: select_probability(row), axis=1)

        # Add columns churn, probability, time, and model used to uploaded df and form df

        def add_columns(df):
            df[convert_string(df, 'churn')] = predictions
            df[convert_string(df, 'probability_(%)')] = probabilities.ravel()
            df[convert_string(df, 'time_of_prediction')
               ] = dt.datetime.now()
            df[convert_string(df, 'model_used')
               ] = st.session_state['selected_model']

            return df

        # Form df if search customer is true or df from Uploaded data
        if search_customer:
            df = add_columns(df)

            df.to_csv(HISTORY_FILE, mode='a', header=not os.path.isfile(
                HISTORY_FILE))  # Save only known customers

        else:
            df = add_columns(df_uploaded)  # Raw, No cleaning

        # Store df with prediction results in session state
        st.session_state['bulk_prediction_df'] = df

    return df


def search_customer_form(pipeline, encoder):
    _, df_test = get_test_data()

    customer_ids = df_test['customer_id'].unique().tolist()+['']
    if st.session_state['sidebar'] == 'single_prediction':
        with st.form('customer_id_form'):
            col1, _ = st.columns(2)
            with col1:
                st.write('#### Customer Id 👨🏻‍👩🏻‍👦🏻‍👦🏻')
                st.selectbox(
                    'Search a customer', options=customer_ids, index=len(customer_ids)-1, key='search_customer_id')
            st.form_submit_button('Predict', type='primary', on_click=make_prediction, kwargs=dict(
                pipeline=pipeline, encoder=encoder))
    else:
        with st.form('customer_id_bulk_form'):
            col1, _ = st.columns(2)
            with col1:
                st.write('#### Customer Id 👨🏻‍👩🏻‍👦🏻‍👦🏻')
                st.multiselect(
                    'Search a customer', options=customer_ids, default=None, key='customer_id_bulk')
            st.form_submit_button('Predict', type='primary', on_click=make_predictions, kwargs=dict(
                pipeline=pipeline, encoder=encoder))


def gen_random_customer_id():
    numbers = ''.join(random.choices(string.digits, k=4))
    letters = ''.join(random.choices(string.ascii_uppercase, k=5))
    return f"{numbers}-{letters}-random"


def manual_customer_form(pipeline, encoder):
    with st.form('input_features_form'):

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write('##### Customer Demographics 🌍')
            st.text_input(
                'customer ID', value=gen_random_customer_id(), key='manual_customer_id')
            st.selectbox(
                'Gender', options=['Male', 'Female'], key='gender')
            st.selectbox('Senior Citizen', options=[
                'Yes', 'No'], key='senior_citizen')
            st.selectbox('Partner', options=[
                'Yes', 'No'], key='partner')
            st.selectbox('dependents', options=[
                'Yes', 'No'], key='dependents')

        with col2:
            st.write('##### Services Subscribed 🟢')
            st.selectbox('Phone Service', options=[
                'Yes', 'No'], key='phone_service')
            st.selectbox('Multiple Lines', options=[
                'Yes', 'No'], key='multiple_lines')
            st.selectbox('Internet Service', options=[
                'DSL', 'Fiber optic', 'No'], key='internet_service')
            st.number_input(
                'Tenure (months)', min_value=0, max_value=120, step=1, key='tenure')
            st.selectbox('Contract', options=[
                'Month-to-month', 'One year', 'Two year'], key='contract')

        with col3:
            st.write('##### Internet related services 📶')
            st.selectbox('Online Security', options=[
                'Yes', 'No'], key='online_security')
            st.selectbox('Online Backup', options=[
                'Yes', 'No'], key='online_backup')
            st.selectbox('Device Protection', options=[
                'Yes', 'No'], key='device_protection')
            st.selectbox('Tech Support', options=[
                'Yes', 'No'], key='tech_support')
            st.selectbox('Streaming TV', options=[
                'Yes', 'No'],  key='streaming_tv')
            st.selectbox('Streaming Movies', options=[
                'Yes', 'No'], key='streaming_movies')
        with col4:
            st.write('##### Billing and Payment 💵')
            st.selectbox('Paperless Billing', options=[
                'Yes', 'No'], key='paperless_billing')
            st.selectbox('Payment Method', options=[
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], key='payment_method')
            st.number_input(
                'Monthly Charges ($)', min_value=0.0, format="%.2f", step=1.00, key='monthly_charges')
            st.number_input(
                'Total Charges ($)', min_value=0.0, format="%.2f", step=1.00, key='total_charges')

        st.form_submit_button('Predict', type='primary', on_click=make_prediction, kwargs=dict(
            pipeline=pipeline, encoder=encoder))


def do_single_prediction(pipeline, encoder):
    if st.session_state.get('search_customer', False):
        search_customer_form(pipeline, encoder)

    else:
        manual_customer_form(pipeline, encoder)


def show_prediction():
    final_prediction = st.session_state.get('prediction', None)
    final_probability = st.session_state.get('probability', None)

    if final_prediction is None:
        st.markdown('#### Prediction will show below! 🧠')
        st.divider()
    else:
        st.markdown('#### Prediction! 🧠')
        st.divider()
        if final_prediction.lower() == 'yes':
            st.toast("Retention o'clock!", icon='🧲')
            message = f'It is **{
                final_probability:.2f}%** likely that the customer will **leave.**'
            st.warning(message, icon='😞')
            time.sleep(5)
            st.toast(message)
        else:
            st.balloons()
            st.toast("Loyalty o'clock!", icon='🤝')
            message = f'The customer will **stay** with a likelihood of **{
                final_probability:.2f}%**.'
            st.success(message, icon='😊')
            time.sleep(5)
            st.toast(message)

    # Set prediction and probability to None
    st.session_state['prediction'] = None
    st.session_state['probability'] = None


@st.cache_data(show_spinner=False)
def convert_df(df: pd.DataFrame):
    return df.to_csv(index=False)


def bulk_upload_widget(pipeline, encoder):
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel File", type=['csv', 'xls', 'xlsx'])

    uploaded = uploaded_file is not None

    upload_bulk_predict = st.button('Predict', type='primary',
                                    help='Upload a csv/excel file to make predictions', disabled=not uploaded, key='upload_bulk_predict')
    df = None
    if upload_bulk_predict and uploaded:
        df_test_raw, _ = get_test_data()
        # Uploadfile is a "file-like" object is accepted
        try:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                df = pd.read_excel(uploaded_file)

            df_columns = set(df.columns)
            df_test_columns = set(df_test_raw.columns)
            df_schema = df.dtypes
            df_test_schema = df_test_raw.dtypes

            if df_columns != df_test_columns or not df_schema.equals(df_test_schema):
                df = None
                raise Exception
            else:
                # Clean dataframe
                janitor = Janitor()
                df_clean = janitor.clean_dataframe(df)

                df = make_predictions(
                    pipeline, encoder, df_uploaded=df, df_uploaded_clean=df_clean)

        except Exception:
            st.subheader('Data template')
            data_template = df_test_raw[:3]
            st.dataframe(data_template)
            csv = convert_df(data_template)
            message_1 = 'Upload a valid csv or excel file.'
            message_2 = f"{message_1.split(
                '.')[0]} with the columns and schema of the above data template."
            icon = '😞'
            st.toast(message_1, icon=icon)
            st.download_button(
                label='Download template',
                data=csv,
                file_name='Data template.csv',
                mime="text/csv",
                type='secondary',
                key='download-data-template'
            )
            st.info('Download the above template for use as a baseline structure.')
            # Display explander to show the data dictionary
            with st.expander("Expand to see the data dictionary", icon="💡"):
                st.subheader("Data dictionary")
                st.markdown(markdown_table_all)
            st.warning(message_2, icon=icon)

    return df


def do_bulk_prediction(pipeline, encoder):
    if st.session_state.get('search_customer', False):
        search_customer_form(pipeline, encoder)

    else:
        # File uploader
        bulk_upload_widget(pipeline, encoder)


def show_bulk_predictions(df: pd.DataFrame):
    if df is not None:
        st.subheader("Bulk predictions of customer churn 🔮", divider=True)
        st.dataframe(df.astype(str))

        csv = convert_df(df)
        message = 'The predictions are ready for download.'
        icon = '⬇️'
        st.toast(message, icon=icon)
        st.info(message, icon=icon)
        st.download_button(
            label='Download predictions',
            data=csv,
            file_name='Bulk prediction.csv',
            mime="text/csv",
            type='secondary',
            key='download-bulk-prediction'
        )

        # Set bulk prediction df to None
        st.session_state['bulk_prediction_df'] = None


def sidebar(sidebar_type: str):
    return st.session_state.update({'sidebar': sidebar_type})


def main():
    st.title("Predict Customer Churn 🤖")

    st.sidebar.toggle("Looking for a customer?", value=st.session_state.get(
        'search_customer', False), key='search_customer')

    # tab1, tab2 = st.tabs(['🧠 Predict', '🔮 Bulk predict'])
    chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id=1, title='🧠 Predict', description=''),
        stx.TabBarItemData(id=2, title='🔮 Bulk predict', description=''),
    ], default=1)

    pipeline, encoder = select_model()

    if chosen_id == '1':
        sidebar('single_prediction')
        do_single_prediction(pipeline, encoder)
        show_prediction()

    else:
        sidebar('bulk_prediction')
        df_with_predictions = do_bulk_prediction(pipeline, encoder)
        if df_with_predictions is None:
            df_with_predictions = st.session_state.get(
                'bulk_prediction_df', None)
        show_bulk_predictions(df_with_predictions)


if __name__ == '__main__':
    main()
