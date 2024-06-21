import os

# Paths
BASE_DIR = '../'
DATA = os.path.join(BASE_DIR, 'data/')
SECOND_FILE = os.path.join(DATA, 'raw/LP2_Telco-churn-second-2000.csv')
TRAIN_FILE = os.path.join(DATA, 'raw/df_train.csv')
TRAIN_FILE_CLEANED = os.path.join(DATA, 'cleaned/df_train.csv')
TEST_FILE = os.path.join(DATA, 'raw/Telco-churn-last-2000.xlsx')
HISTORY = os.path.join(DATA, 'history/')
MODELS = os.path.join(BASE_DIR, 'models/')
ENCODER = os.path.join(MODELS, 'enc/')

# Urls
SECOND_FILE_URL = "https://raw.githubusercontent.com/Azubi-Africa/Career_Accelerator_LP2-Classifcation/main/LP2_Telco-churn-second-2000.csv"
TEST_FILE_URL = 'https://github.com/D0nG4667/telco_customer_churn_prediction/raw/main/data/untouched/Telco-churn-last-2000.xlsx'
