import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer
import pandas as pd

from config.settings import HISTORY_FILE, HISTORY_FILE_URL
from includes.logo import logo
from includes.footer import footer


# Set page configuration
st.set_page_config(
    page_title='History Page',
    page_icon='🕰️',
    layout="wide",
    initial_sidebar_state='auto'
)

# Use app logo
logo()


@st.cache_data(show_spinner="Getting history of predictions...")
def get_history_data():
    try:
        df_history = pd.read_csv(
            HISTORY_FILE_URL, index_col=0)
    except Exception:
        df_history = pd.read_csv(
            HISTORY_FILE, index_col=0)

    df_history['time_of_prediction'] = [timestamps[0]
                                        for timestamps in df_history['time_of_prediction'].str.split('.')[0:]]

    df_history['time_of_prediction'] = pd.to_datetime(
        df_history['time_of_prediction'])

    return df_history


def main():
    st.title("Prediction History 🕰️")

    df_history = get_history_data()

    df_history_explorer = dataframe_explorer(df_history, case=False)

    st.dataframe(df_history_explorer)

    # Add footer
    footer()


if __name__ == '__main__':
    main()
