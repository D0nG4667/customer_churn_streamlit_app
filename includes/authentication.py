import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from config.settings import CONFIG_YAML

# CONFIG_YAML = "./config/config.yaml"


def get_authenticator():
    with open(CONFIG_YAML) as file:
        config = yaml.load(file, Loader=SafeLoader)

        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
            config['preauthorized']
        )

        name, authentication_status, username = authenticator.login(
            location='sidebar')

        return name, authentication_status, username, authenticator


def default_credentials(error=False):
    if error:
        st.error('You may have entered wrong credentials', icon='🚨')

    st.info('Login to get use this data app', icon='🔓')

    st.markdown(f"""
        **Test Account**

        Username:
    """)
    st.code(st.secrets.local.authenticator.USER)
    st.markdown(f"""
        Password:
    """)
    st.code(st.secrets.local.authenticator.PASS)


def logout_button(authenticator):
    return authenticator.logout(location='sidebar') if st.session_state['authentication_status'] else default_credentials()


def is_login(authenticator):
    return default_credentials(error=True) if not st.session_state['authentication_status'] else logout_button(authenticator)


def add_authentication():
    name, authentication_status, username, authenticator = get_authenticator()
    if authentication_status is None:
        authentication_status = False
    if authentication_status ^ st.session_state.get('FormSubmitter:Login-Login', False):
        is_login(authenticator)
    else:
        default_credentials()

    return name, authentication_status, username, authenticator


"""
    Usage:
    if username and name and authentication_status:
        st.switch_page
        st.write(username)
        Do something
    
"""