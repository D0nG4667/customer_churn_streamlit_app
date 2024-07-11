import streamlit as st
import pandas as pd

from config.settings import TRAIN_FILE_CLEANED, TRAIN_FILE_CLEANED_URL

from includes.eda import eda
from includes.kpi import kpi
from includes.func_in_pipeline import *
from includes.model_explainer import model_explainer
from includes.logo import logo
from includes.authentication import add_authentication
from includes.footer import footer


st.set_page_config(
    page_title='Dashboard Page',
    page_icon='📈',
    layout="wide",
    initial_sidebar_state='auto'
)

# Use app logo
logo()


@st.cache_data(show_spinner="Getting data for visualisation...")
def get_train_data():
    try:
        df_clean = pd.read_csv(TRAIN_FILE_CLEANED_URL)
    except Exception:
        df_clean = pd.read_csv(TRAIN_FILE_CLEANED)

    # Some house keeping, explicitly fill missing values in categorical columns as "No"
    categoricals = df_clean.select_dtypes(exclude=['number']).columns.tolist()
    df_clean[categoricals] = df_clean[categoricals].fillna("No")

    return df_clean


def main():
    st.title("Custormer Churn Dashboard 📈")

    df_clean = get_train_data()

    if "toast" not in st.session_state:
        st.toast("Data loaded successfully!", icon="✔️")
        st.session_state["toast"] = True

    _, col2 = st.columns(2)
    with col2:
        st.selectbox("Select dashboard type", options=[
                     "EDA", "KPI", "Model Explainer"], key="dashboard_type", index=1)

    target = 'churn'

    if st.session_state["dashboard_type"] == "EDA":
        eda(df_clean, target)
    elif st.session_state["dashboard_type"] == "KPI":
        kpi(df_clean, target)
    else:
        model_explainer(df_clean)


if __name__ == "__main__":
    add_authentication(main, footer)
