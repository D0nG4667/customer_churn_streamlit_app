import streamlit as st

from config.settings import LOGO


def logo():
    return st.logo(image=LOGO)

