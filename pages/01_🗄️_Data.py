import os
import time
import urllib

import pandas as pd
import streamlit as st
import extra_streamlit_components as stx
from sqlalchemy import create_engine

from config.settings import DATA, FIRST_FILE, SECOND_FILE, SECOND_FILE_URL, TRAIN_FILE, TRAIN_FILE_CLEANED, TEST_FILE, TEST_FILE_URL
from includes.logo import logo
from includes.markdown import *
from includes.janitor import Janitor


st.set_page_config(
    page_title='Data Page',
    page_icon='🗄️',
    layout="wide"
)

# Use app logo
logo()


@st.cache_resource(show_spinner="Connecting to database...")
def create_connection():
    connection_string = f"DRIVER={{SQL Server}};SERVER={st.secrets['server']};DATABASE={
        st.secrets['database']};UID={st.secrets['username']};PWD={st.secrets['password']};MARS_Connection=yes;MinProtocolVersion=TLSv1.2;"
    # Encode the connection string to be used with SQLAlchemy
    params = urllib.parse.quote_plus(connection_string)
    # Create the SQLAlchemy engine
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    return engine


@st.cache_data(show_spinner="Saving datasets...")
def save_dataset(df, filepath, csv=True):
    def save(df, file):
        return df.to_csv(file, index=False) if csv else df.to_excel(file, index=False)

    def read(file):
        return pd.read_csv(file) if csv else pd.read_excel(file)

    def same_dfs(df, df2):
        return df.equals(df2)

    if not os.path.isfile(filepath):  # Save if file does not exists
        save(df, filepath)
    else:  # Save if data are not same
        df_old = read(filepath)
        if not same_dfs(df, df_old):
            save(df, filepath)


@st.cache_data(show_spinner="Running query to get first data...")
def get_first_data():
    try:
        connection = create_connection()
        sql_query = f"SELECT * FROM {st.secrets['table']}"
        first_dataset = pd.read_sql(sql_query, connection)
        save_dataset(first_dataset, FIRST_FILE)
    except Exception:
        first_dataset = pd.read_csv(FIRST_FILE)
    return first_dataset


@st.cache_data(show_spinner="Getting second data...")
def get_second_data():
    try:
        second_dataset = pd.read_csv(SECOND_FILE_URL)  # Online
        save_dataset(second_dataset, SECOND_FILE)  # Save offline
    except Exception:
        second_dataset = pd.read_csv(SECOND_FILE)  # Use offline
    return second_dataset


# @st.cache_data(show_spinner="Cleaning data...")
def get_clean_data(df):
    try:
        janitor = Janitor()
        # Apply all cleaning procedure in sequence
        df_clean = janitor.clean_dataframe(df)
        save_dataset(df_clean, TRAIN_FILE_CLEANED)  # Save offline
    except Exception:
        df_clean = pd.read_csv(TRAIN_FILE_CLEANED)  # Use offline
    return df_clean


@st.cache_data(show_spinner="Getting testing data...")
def get_test_data():
    try:
        test_dataset = pd.read_excel(TEST_FILE_URL)
        save_dataset(test_dataset, TEST_FILE, csv=False)
    except Exception:
        test_dataset = pd.read_excel(TEST_FILE)
    return test_dataset


@st.cache_data(show_spinner="Getting train data...")
def get_train_data(first_dataset, second_dataset):
    df_train = pd.concat([first_dataset, second_dataset], ignore_index=True)
    save_dataset(df_train, TRAIN_FILE)
    return df_train


def get_all_data(progress_bar):
    first_dataset = get_first_data()
    progress_bar.progress(40)
    second_dataset = get_second_data()
    progress_bar.progress(55)
    df_test = get_test_data()
    progress_bar.progress(65)
    df_train = get_train_data(first_dataset, second_dataset)
    df_train_clean = get_clean_data(df_train)
    progress_bar.progress(75)
    return df_test, df_train, df_train_clean


@st.cache_data(show_spinner="Filtering data...")
def filter_columns_and_markdown(df, category, view):
    def filter_markdown(markdown, markdown_cleaned):
        return markdown_cleaned if view == 'cleaned' else markdown
    if category == "Numerical Columns":
        filtered_df = df.select_dtypes(include="number")
        filtered_markdown_table = filter_markdown(
            markdown_table_num, markdown_table_num_cleaned)
    elif category == "Categorical Columns":
        filtered_df = df.select_dtypes(exclude="number")
        filtered_markdown_table = filter_markdown(
            markdown_table_cat, markdown_table_cat_cleaned)
    else:
        filtered_df = df
        filtered_markdown_table = filter_markdown(
            markdown_table_all, markdown_table_all_cleaned)
    return filtered_df, filtered_markdown_table


def tab_contents(df, view='raw'):
    filtered_data, filtered_markdown_table = filter_columns_and_markdown(
        df, st.session_state.category, view)
    st.dataframe(filtered_data.astype(str))
    filename = f"{st.session_state.category} ({view})"
    extension = ".csv"
    file = filename+extension
    filepath = os.path.join(DATA, file)
    save_dataset(filtered_data, filepath)

    with open(filepath, "rb") as fp:
        st.download_button(
            label=f"Download {filename.lower()}",
            data=fp,
            file_name=file,
            mime="text/csv",
            key=f"download-data-{view}"
        )

    # Display explander to show feature descriptions
    with st.expander("Expand to learn about the features", icon="💡"):
        st.subheader("Data dictionary")
        st.write(st.session_state["category"])
        st.markdown(filtered_markdown_table)


def main():
    st.title("Proprietory Data from Vodafone 🗄️")

    # Create a progress bar to let user know data is loading
    progress_bar = st.progress(10)

    # Get data for viewing
    df_test, df_train, df_train_clean = get_all_data(progress_bar)

    for percentage_completed in range(25):
        time.sleep(0.005)
        progress_bar.progress(75 + percentage_completed + 1)

    progress_bar.empty()
    st.toast("Data was loaded successfully!")

    # Initialize the session state for categories
    if "category" not in st.session_state:
        st.session_state["category"] = "All Columns"

    # Create the tabs
    # tab1, tab2, tab3 = st.tabs(["📄 Raw", "✨ Cleaned", "📜 Test"]) # No session state with st inbuilt tabs
    chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id=1, title='📄 Raw', description=''),
        stx.TabBarItemData(id=2, title='✨ Cleaned', description=''),
        stx.TabBarItemData(id=3, title='📜 Test', description=''),
    ], default=1)

    _, col2 = st.columns(2)
    with col2:
        st.selectbox("Select Specific Features", options=[
                     "All Columns", "Numerical Columns", "Categorical Columns"], key="category")

    # Show the tabs
    if chosen_id == '1':
        st.subheader("Data view of the raw dataset")
        tab_contents(df_train, view='raw')
    elif chosen_id == '2':
        st.subheader("Data view of the cleaned dataset")
        if not st.session_state.get('snow', False):
            st.snow()
            st.session_state['snow'] = True
        tab_contents(df_train_clean, view='cleaned')
    else:
        st.subheader("Data view of the test dataset")
        tab_contents(df_test, view='test')


if __name__ == "__main__":
    main()
