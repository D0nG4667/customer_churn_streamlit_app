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

num_uni_biv_insight = r"""
    #### Key Insights
    `Tenure:` Analysis of customer tenure reveals a diverse pattern of engagement with the company. The majority of customers exhibit relatively short tenure, with many staying for less than 10 months. However, there is an interesting outlier observed, indicating a small but notable spike in customer loyalty, with some individuals remaining with the company for up to 72 months.

    `Monthly Charges:` Examination of monthly charges illustrates a right-skewed distribution, with a significant portion of customers being charged around \$70.55 monthly, as indicated by the median. However, there is substantial variability in charges beyond this point, ranging from \$18.40 to \$118.65. This variability suggests diverse pricing plans or additional services catering to different customer needs and preferences. Notably, most of the customers who churn have monthly charges above \$70.00.

    `Total Charges:` The analysis of total charges reveals a concentration within the range of \$18.80 to \$2000.00. This indicates that the majority of customers have accumulated charges within this bracket. However, there are also notable instances of higher total charges up to \$8,670.10, suggesting variations in usage, additional services, or other factors influencing overall expenditure.
    
    The correlation heatmap reveals that `tenure` has a strong positive correlation (0.826) with `total_charges` while its correlation (0.241) with `monthly_charges` is weak. Although, `monthly_charges` and `total_charges` have a strong positive correlation (0.647) but less than (0.826). 
"""

num_mul_violin_insight = """
    #### Key Insight
    Customers retention implying longer tenure is influenced by automatic payment methods- bank transfer and credit card. Customers who make payments automatically are less likely to churn compared to those who use check payment methods- electronic and mailed.
"""

num_mul_pca_insight = """
    #### Key Insights
    The PCA plot above visualizes the relationships between customers churn based on their tenure, monthly charges, and total charges. The plot displays the first two principal components, which capture the most significant sources of variance in the dataset.

    `Direction of Data Points:` Each point on the plot represents an individual customer. The direction and distance between points reflect similarities or differences in their tenure and charges.

    `Clusters and Patterns:` Clusters or groupings of points suggest similarities among customers. For instance, a dense cluster in one area of the plot may indicate a group of customers with similar tenure and charge characteristics, such as long-term customers with high monthly and total charges.

    `Outliers:` Points that are far from the main cluster(s) may represent outliers—customers with unique characteristics compared to the rest of the dataset. These outliers could be customers with exceptionally high or low charges relative to their tenure.

    `Variance Explained:` The first two components explain a significant portion of the total variance 100.0%, suggesting the visualization of the dataset's structure in two dimensions is effective.   
"""

cat_uni_biv_insight = """    
    ### Key Insights

    `Gender:` Male customers slightly outnumber female customers.

    `Partner:` The proportion of customers with or without partners is approximately equal.

    `Dependents:` There are more customers without dependent members compared to those with dependents.

    `Phone Service:` The majority of customers do not have phone service, outnumbering those who do.

    `Internet Service:` Customers with internet service predominantly opt for DSL or Fiber optic connections.

    `MultipleLines, InternetService, OnlineSecurity, OnlineBackup, TechSupport`: A consistent pattern emerges across these features, with most customers preferring not to access these features.

    `StreamingMovies and StreamingTV:` Similar barplots indicate an equal preference among customers for having or not having these services.

    `Contract:` Customers generally prefer month-to-month contracts over longer-term options such as two-year or one-year contracts.

    `Paperless Billing:` The majority of customers prefer paperless billing, utilizing various forms of banking transactions, with Electronic Check being the most common.

    Churn Analysis- Customers more likely to churn:
    - Those without partners.
    - Those without dependents.
    - Those with phone service.
    - Those using fiber optic internet service.
    - Those not subscribing to extra services like Online Backup or Online Security.
    - Those on a month-to-month contract basis.
    - Those using Electronic Check as their payment method.

    ### Recommendations:

    - Vodafone could enhance the electronic check payment method experience to ensure convenience and ease of use for customers, potentially reducing churn rates.
    - Consider improve customer experience and offer discount on family plans, phone services and cross selling other services with online security and backup.
    - More investigation into customer experience with fiber optic connections should be engaged. A questionnaire or survey approach may be a good start.

"""

cat_mul_insight = """
    ### Key Insights

    `Significant Variables:` The majority of the variables exhibit a p-value of 0.00, indicating a significant association with churn. These variables include contract type, online security, tech support, dependents, online backup, senior citizen status, partner status, paperless billing, payment method, device protection, and internet service.

    `Non-Significant Variables:` Variables such as streaming TV, streaming movies, multiple lines, phone service, and gender have p-values above the typical significance threshold of 0.05. While streaming TV, streaming movies, and multiple lines have relatively low p-values, indicating some association with churn, they may not be as influential as the other variables in predicting churn.

    #### Impact on Modeling Churn Prediction:
    `Significant Variables:` Variables with significant p-values are crucial for modeling churn prediction as they provide valuable information about customer behavior and preferences. The variables will be incorporated into the churn prediction model to improve its performance in identifying customers at risk of churn.

    `Non-Significant Variables:` While non-significant variables may still have some predictive power, their impact on the overall churn prediction model may be limited. It's essential to prioritize variables with significant associations with churn when building the predictive model to ensure its robustness and reliability. Considerations will be made to create new features from these non-significant features.
"""


log_reg_insight = """
    ### `Understanding Feature Importances in Customer Churn Prediction`

    ### Overview
    We leveraged logistic regression, our best-performing model, to discern the most influential factors predicting customer behavior within our dataset. The coefficients extracted from the model, denoted as "feature importances," elucidate the impact of each variable on the likelihood of customer actions, such as churn or retention.

    ### Key Findings

    **1. Tenure**: 
    - **Impact**: This feature exhibits the most substantial negative impact on the outcome (-2.02).
    - **Interpretation**: Longer tenure diminishes the probability of churn, suggesting that established customers are more inclined to remain with the service.

    **2. Contract Type**:
    - **Month-to-Month Contracts**: Positively correlated with the outcome (+0.647), indicating higher volatility or turnover among short-term customers.
    - **Two-Year Contracts**: Displays a significant negative coefficient (-1.68), signifying enhanced customer retention and stability.

    **3. Internet Service**:
    - **Fiber Optic Services**: Positively influences the outcome (+1.90), potentially reflecting heightened expectations or distinct service experiences.
    - **No Internet Service**: Exhibits a negative coefficient (-1.36), lowering the likelihood of churn, possibly due to reduced engagement with services.

    **4. Billing and Payment Methods**:
    - **Electronic Checks**: Positively associated with the outcome (+0.25), suggesting a potential link to more transient or less satisfied customer segments.
    - **Mailed Checks**: Shows a negative coefficient (-0.10), albeit with lesser significance, indicating a different customer behavior pattern.

    **5. Add-On Services**:
    - Features such as **security services**, **call services**, and **streaming services** display varying impacts. Their presence tends to either increase or decrease the likelihood of churn, underscoring their influence on customer satisfaction and retention.

    ### Implications and Recommendations

    - **Customer Retention**: Strengthen retention strategies by enhancing service offerings for long-tenure customers, particularly those with stable contract setups like two-year agreements.
    - **Service Improvement**: Investigate the significant impact of fiber optic services on customer behavior, focusing on improving service quality or customer support for these users.
    - **Payment Flexibility**: Consider promoting automatic payment methods, which appear to be associated with more stable customer behavior, potentially enhancing overall customer satisfaction and retention.
    - **Targeted Marketing**: Tailor marketing strategies to address the specific needs of different customer segments, particularly focusing on those with month-to-month contracts or using electronic checks.

    **The most important features for predicting churn are whether a customer has fibre optic internet service, a contract term of two years and tenure. Other features such as monthly charges, total charges, contract of one year, electronic check payment method, whethether a customer has streaming movies, tech support and online security services are also important although around half the most important features.**
"""
