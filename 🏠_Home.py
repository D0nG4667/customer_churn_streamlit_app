import streamlit as st

from includes.logo import logo
from includes.authentication import add_authentication
from includes.footer import footer


# Set page configuration
st.set_page_config(
    page_title='Home Page',
    page_icon='🏠',
    layout="wide",
    initial_sidebar_state='auto'
)

# Use app logo
logo()


def main():
    st.header(
        '🚀 Elevating ML Models from Development to Real-world Impact', divider=True)

    # Introduction
    st.write("""
            <p style="font-size: 3em; float: left; line-height: 0.9;">V</p>odafone seeks to enhance its customer retention strategies by predicting customer churn using machine learning models. 
            This project, leveraging the Streamlit framework, outlines the creation of a data application to deploy predictive models with a user-friendly interface. </br>


            #### Why Model Deployment?
            Incorporating customer churn prediction into Vodafone's strategy is vital. 
            By predicting which customers are likely to leave, Vodafone can implement targeted retention efforts. 
            Deploying these models through Streamlit empowers stakeholders with actionable insights, seamlessly integrating into existing business environments.             
             """, unsafe_allow_html=True)

    # About the app
    st.write('#### About This App')
    st.write("""   
        Our intelligent application empowers stakeholders to predict customer churn effectively. 
        By integrating machine learning models into a user-friendly and intuitive interface, the app provides real-time insights, helping business stakeholders implement targeted retention strategies. 
        This user-friendly solution enhances decision-making and improves customer retention efforts.
    """)

    # Create columns for Key Features and User Benefits
    col1, col2 = st.columns(2)

    # Key Features
    with col1:
        st.write("#### Key Features")
        st.write("""
                 
            - **Data:** A page that allows users to view the raw, clean and test data across the various tabs. It includes a download csv feature. 
            The data cleaning logic and pipeline is in the code base of this page.
            - **Dashboard:** It contains visualizations of the dataset used to train the models. 
            You can select between the 3 different types using a drop down menu. An **EDA** dashboard, an analytics **KPIs** dashboard with filtering capabilities and 
            a **model explainer** dashboard with Confusion matrix, AUC ROC Curves and a feature importances visualization to understand the drivers of customer churn for the leading model.        
            - **Predict:** Contains a list of models in a drop down with corresponding pipelines used to predict customer churn. There are two tabs namely predict and bulk predict for single and bulk prediction(s) respectively. 
            The bulk predict allow users to upload of csv or excel files with same schema as the data used for training and testing. 
            A toggle button allow users to make predictions by way of searching a particular customer or several customers in the predict and bulk predict tabs respectively.  
            - **History:** This page contains table view of the past predictions made including their churn, probability, model used and time stamp.
          
        """)

    # User Benefits
    with col2:
        st.write("#### User Benefits")
        st.write("""
        - **Decision-driven decisions:** Make data-driven decisions effortlessly harnessing the power of a data app that integrates analytics, machine learning and predictions.
        - **Improve Customer Retention:** Identify at-risk customers and implement proactive retention strategies.
        - **Optimize Marketing Strategies:** Customize marketing efforts to effectively target potential churners.
        - **Enhance Business Performance:** Lower churn rates and boost customer lifetime value.
        """)

    # Create two columns for Machine Learning Integration and How to run application
    col3, col4 = st.columns(2)

    # User Benefits
    with col3:
        st.write("#### Machine Learning Integration")
        st.write("""
        - **Nine(9) models:** You have access to select between 9 models for prediction.
        - **Save predictions:** Predictions are automatically saved at the backend and show up on the history page.
        - **Download bulk predictions:** Bulk predictions have an additional download as csv feature.
        - **Probability and likelihood:** All prediction show their likelihoods expressed as a percentage.
        """)

    # How to run the application is a development environment
    with col4:
        st.write("#### How to run the application")
        with st.container(border=True):
            st.code("""
                # Create the conda environemnt
                conda env create -f streamlit_environment.yml
                
                # Activate the cond a environment
                conda activate streamlit_env
                
                # Run the app
                streamlit run 🏠_Home.py            
            """)

    # Create two columns for Live Demo and Need Consultation?
    col5, col6 = st.columns(2)

    # Live Demo
    with col5:
        st.write("#### Live Demo")
        st.write("##### [Watch Demo Video](https://youtu.be/M5w4wr9lI-Y)")

    # Need Consultation?
    with col6:
        st.write("#### Need Consultation?")
        st.write("Have questions or need insights? Connect with me on LinkedIn, drop me an email, or explore my articles on Medium to dive deeper. Let's create insights one data point at a time.")
        st.write("""
                 [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/dr-gabriel-okundaye)&nbsp; 
                 [![Email](https://img.shields.io/badge/Email-Contact-blue)](mailto:gabriel.okundaye@statogale.com)&nbsp; 
                 [![Medium](https://img.shields.io/badge/Medium-Read-blue)](https://medium.com/@gabriel007okuns)&nbsp;
        """)


if __name__ == "__main__":
    with st.sidebar:
        name, authentication_status, username, authenticator = add_authentication()

    if st.session_state.get('username') and st.session_state.get('name') and st.session_state.get('authentication_status'):
        main()
    else:
        st.info('### 🔓 Login to access this data app')
        footer()
