markdown_table_num = """
| Column Names|Description| Data Type|
|-------------|-----------|----------|
|SeniorCitizen|Whether a customer is a senior citizen or not|int64|
|tenure|Number of months the customer has stayed with the company|int64|
|MonthlyCharges|The amount charged to the customer monthly|float64|
"""

markdown_table_cat = """
| Column Names|Description| Data Type|
|-------------|-----------|----------|
|CustomerID|The id of the customer|object|
|gender|Whether the customer is a male or a female|object|
|Partner|Whether the customer has a partner or not (Yes, No)|object|
|Dependents|Whether the customer has dependents or not (Yes, No)|object|
|PhoneService|Whether the customer has a phone service or not (Yes, No)|object|
|MultipleLines|Whether the customer has multiple lines or not|object|
|InternetService|Customer's internet service provider (DSL, Fiber Optic, No)|object|
|OnlineSecurity|Whether the customer has online security or not (Yes, No, No Internet)|object|
|OnlineBackup|Whether the customer has online backup or not (Yes, No, No Internet)|object|
|DeviceProtection|Whether the customer has device protection or not (Yes, No, No internet service)|object|
|TechSupport|Whether the customer has tech support or not (Yes, No, No internet)|object|
|StreamingTV|Whether the customer has streaming TV or not (Yes, No, No internet service)|object|
|StreamingMovies|Whether the customer has streaming movies or not (Yes, No, No Internet service)|object|
|Contract|The contract term of the customer (Month-to-Month, One year, Two year)|object|
|PaperlessBilling|Whether the customer has paperless billing or not (Yes, No)|object|
|PaymentMethod|The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))|object| 
|TotalCharges|The total amount charged to the customer|object|
|Churn|Whether the customer churned or not (Yes or No)|object|
"""

markdown_table_all = """
| Column Names|Description| Data Type|
|-------------|-----------|----------|
|CustomerID|The id of the customer|object|
|gender|Whether the customer is a male or a female|object|
|SeniorCitizen|Whether a customer is a senior citizen or not|int64|
|Partner|Whether the customer has a partner or not (Yes, No)|object|
|Dependents|Whether the customer has dependents or not (Yes, No)|object|
|tenure|Number of months the customer has stayed with the company|int64|
|PhoneService|Whether the customer has a phone service or not (Yes, No)|object|
|MultipleLines|Whether the customer has multiple lines or not|object|
|InternetService|Customer's internet service provider (DSL, Fiber Optic, No)|object|
|OnlineSecurity|Whether the customer has online security or not (Yes, No, No Internet)|object|
|OnlineBackup|Whether the customer has online backup or not (Yes, No, No Internet)|object|
|DeviceProtection|Whether the customer has device protection or not (Yes, No, No internet service)|object|
|TechSupport|Whether the customer has tech support or not (Yes, No, No internet)|object|
|StreamingTV|Whether the customer has streaming TV or not (Yes, No, No internet service)|object|
|StreamingMovies|Whether the customer has streaming movies or not (Yes, No, No Internet service)|object|
|Contract|The contract term of the customer (Month-to-Month, One year, Two year)|object|
|PaperlessBilling|Whether the customer has paperless billing or not (Yes, No)|object|
|PaymentMethod|The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))|object| 
|MonthlyCharges|The amount charged to the customer monthly|float64|
|TotalCharges|The total amount charged to the customer|object|
|Churn|Whether the customer churned or not (Yes or No)|object|
"""


markdown_table_num_cleaned = """
| Column Names|Description| Data Type|
|-------------|-----------|----------|
|tenure|Number of months the customer has stayed with the company|int64|
|monthly_charges|The amount charged to the customer monthly|float64|
|total_charges|The total amount charged to the customer|float64|
"""

markdown_table_cat_cleaned = """
| Column Names|Description| Data Type|
|-------------|-----------|----------|
|gender|Whether the customer is a male or a female|object|
|senior_citizen|Whether a customer is a senior citizen or not|object|
|partner|Whether the customer has a partner or not (Yes, No)|object|
|dependents|Whether the customer has dependents or not (Yes, No)|object|
|phone_service|Whether the customer has a phone service or not (Yes, No)|object|
|multiple_lines|Whether the customer has multiple lines or not|object|
|internet_service|Customer's internet service provider (DSL, Fiber Optic, No)|object|
|online_security|Whether the customer has online security or not (Yes, No)|object|
|online_backup|Whether the customer has online backup or not (Yes, No)|object|
|device_protection|Whether the customer has device protection or not (Yes, No)|object|
|tech_support|Whether the customer has tech support or not (Yes, No)|object|
|streaming_tv|Whether the customer has streaming TV or not (Yes, No)|object|
|streaming_movies|Whether the customer has streaming movies or not (Yes, No)|object|
|contract|The contract term of the customer (Month-to-Month, One year, Two year)|object|
|paperless_billing|Whether the customer has paperless billing or not (Yes, No)|object|
|payment_method|The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))|object| 
|churn|Whether the customer churned or not (Yes or No)|object|
"""

markdown_table_all_cleaned = """
| Column Names|Description| Data Type|
|-------------|-----------|----------|
|gender|Whether the customer is a male or a female|object|
|senior_citizen|Whether a customer is a senior citizen or not|object|
|partner|Whether the customer has a partner or not (Yes, No)|object|
|dependents|Whether the customer has dependents or not (Yes, No)|object|
|tenure|Number of months the customer has stayed with the company|int64|
|phone_service|Whether the customer has a phone service or not (Yes, No)|object|
|multiple_lines|Whether the customer has multiple lines or not|object|
|internet_service|Customer's internet service provider (DSL, Fiber Optic, No)|object|
|online_security|Whether the customer has online security or not (Yes, No)|object|
|online_backup|Whether the customer has online backup or not (Yes, No)|object|
|device_protection|Whether the customer has device protection or not (Yes, No)|object|
|tech_support|Whether the customer has tech support or not (Yes, No)|object|
|streaming_tv|Whether the customer has streaming TV or not (Yes, No)|object|
|streaming_movies|Whether the customer has streaming movies or not (Yes, No)|object|
|contract|The contract term of the customer (Month-to-Month, One year, Two year)|object|
|paperless_billing|Whether the customer has paperless billing or not (Yes, No)|object|
|payment_method|The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))|object| 
|monthly_charges|The amount charged to the customer monthly|float64|
|total_charges|The total amount charged to the customer|float64|
|churn|Whether the customer churned or not (Yes or No)|object|
"""
