import streamlit as st
from streamlit_authenticator.authenticate import Authenticate
from streamlit_authenticator.utilities.exceptions import LoginError

from typing import Callable, Tuple, Union

import yaml
from yaml.loader import SafeLoader

from config.settings import CONFIG_YAML


@st.cache_data(show_spinner=False)
def get_config_yaml():
    with open(CONFIG_YAML) as file:
        config = yaml.load(file, Loader=SafeLoader)

    return config


def get_authenticator(config) -> Authenticate:

    authenticator = Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    return authenticator


def show_login_error(authentication_status: Union[bool, None]) -> None:
    if authentication_status is False:  # Show only when status False and not None
        st.error('You may have entered wrong credentials', icon='🚨')


def show_default_credentials(location) -> None:
    placeholder = st.container() if location == 'main' else st.sidebar

    placeholder.info('Login to get use this data app', icon='🔓')

    placeholder.markdown(f"""
        **Test Account**

        Username:
    """)
    placeholder.code(st.secrets.local.authenticator.USER)
    placeholder.markdown(f"""
        Password:
    """)
    placeholder.code(st.secrets.local.authenticator.PASS)


def do_logout(authenticator: Authenticate, location: str):
    st.sidebar.success(f"Howdy, {st.session_state.get('name')}\n" +
                       f"\nEmail: {st.session_state.get('name')}@unicorn.io\n" +
                       f"\nSubscription: **Unlimited**")

    authenticator.logout(location=location)


def login_form(authenticator: Authenticate, location: str) -> Tuple[str, Union[bool, None], str]:
    # name, authentication_status, username
    return authenticator.login(location=location)


def do_login(authenticator: Authenticate, location: str) -> Tuple[str, Union[bool, None], str]:
    name, authentication_status, username = login_form(authenticator, location)
    show_login_error(authentication_status)
    show_default_credentials(location)
    return name, authentication_status, username  # Saved to session_state
    # return tuple(st.session_state[x] for x in ['name', 'authentication_status', 'username'])


def add_authentication(main: Callable = None, footer: Callable = None, location: str = 'sidebar') -> None:
    authenticator = st.session_state.get('authenticator')

    # Use only one Authencticate instance per session state
    if not isinstance(authenticator, Authenticate):
        st.write('no authenticator')
        config = get_config_yaml()
        authenticator = get_authenticator(config)
        st.session_state['authenticator'] = authenticator

    st.write(st.session_state)

    if st.session_state.get('authentication_status') and isinstance(main, Callable):
        do_logout(authenticator, location)
        main()
    elif not st.session_state.get('authentication_status') and isinstance(footer, Callable):
        try:
            do_login(authenticator, location)
        except LoginError as e:
            st.error(e)

        st.info('### 🔓 Login to access this data app')
        footer()

    return authenticator

    # """
    #     Usage:
    #     from includes.footer import footer
    #     from includes.authentication import add_authentication

    # if __name__ == "__main__":
    #     add_authentication(main, footer)
