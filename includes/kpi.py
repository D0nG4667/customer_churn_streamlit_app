import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

import plotly.express as px
import plotly.graph_objects as go


def kpi(df, target):
    """All functions"""

    @st.cache_data(show_spinner=False)
    def filter_data(df, mask):
        return df if (all(mask) or all([not m for m in mask])) else df[mask]

    @st.cache_data(show_spinner=False)
    def kpi_sidebar_mask(df, column, values):
        mask = df[column].isin(values)
        return mask

    def kpi_sidebar(df, categoricals):
        st.sidebar.title("Dashboard filters...")

        select_all = st.sidebar.checkbox("Select all", True)

        masks = []
        for cat in categoricals:
            options = df[cat].unique()
            key = f"{cat}_filter"
            if cat == "customer_id":
                _ = st.sidebar.selectbox(
                    "Looking for a customer?",
                    options,
                    index=None,
                    placeholder="Search a customer...",
                    key=key
                )
                _ = [_] if _ else None
            else:
                _ = st.sidebar.multiselect(cat.replace("_", " ").title(
                ), options, options, key=key) if select_all else st.sidebar.multiselect(cat.replace("_", " ").title(
                ), options, key=key)

            masks.append(kpi_sidebar_mask(df, cat, _ if _ else options))

        # Perform element-wise AND operation across the masks lists to get back a single filter mask
        mask = [all(mask) for mask in zip(*masks)]
        return mask

    @st.cache_data(show_spinner=False)
    def calculate_kpis(df_filtered):
        no_of_customers = len(df_filtered)
        total_revenue = df_filtered['total_charges'].sum()
        churned_customers = (df_filtered['churn'] == 'Yes').sum()
        churn_rate = (churned_customers / no_of_customers) * 100
        monthly_charge = df_filtered['monthly_charges'].median()
        tenure = df_filtered['tenure'].median()

        return [no_of_customers, churned_customers, churn_rate, total_revenue, monthly_charge, tenure]

    def precision(value):
        precision = len(str(value).split('.')[0])
        p, u = (1_000, 'K') if precision < 7 else (1_000_000, 'M')
        value = f'{value/p:,.2f}{u}' if abs(value) >= 1000 else f'{value:,.2f}'

        return value

    def show_kpi_metrics(kpis, kpi_names):
        kpi_metrics = {k: v for k, v in zip(kpi_names, kpis)}

        # Cards
        col1, col2, col3 = st.columns(3)

        with col1:
            no_of_customers = int(kpi_metrics.get('no_of_customers'))
            check_delta = 'delta_1' in st.session_state
            if check_delta:
                delta_1 = no_of_customers - st.session_state['delta_1']

            st.metric(label="No of customers 👫",
                      value=no_of_customers, delta=delta_1 if check_delta else None)
            st.session_state['delta_1'] = no_of_customers

        with col2:
            churned_customers = int(kpi_metrics.get('churned_customers'))
            check_delta = 'delta_2' in st.session_state
            if check_delta:
                delta_2 = churned_customers - st.session_state['delta_2']

            st.metric(label=f"No of customers (churned) 🏃‍♂️",
                      value=churned_customers, delta=delta_2 if check_delta else None)
            st.session_state['delta_2'] = churned_customers

        with col3:
            churn_rate = round(float(kpi_metrics.get('churn_rate')), 1)
            check_delta = 'delta_3' in st.session_state
            if check_delta:
                delta_3 = round(churn_rate - st.session_state['delta_3'], 1)

            st.metric(label="Churn rate 📉",
                      value=f"{churn_rate}%", delta=delta_3 if check_delta else None)
            st.session_state['delta_3'] = churn_rate

        col4, col5, col6 = st.columns(3)

        with col4:
            total_revenue = round(float(kpi_metrics.get('total_revenue')), 2)
            check_delta = 'delta_4' in st.session_state
            if check_delta:
                delta_4 = total_revenue - st.session_state['delta_4']

            st.metric(label="Total revenue",
                      value=f"${precision(total_revenue)}", delta=precision(delta_4) if check_delta else None)
            st.session_state['delta_4'] = total_revenue

        with col5:
            monthly_charge = round(float(kpi_metrics.get('monthly_charge')), 2)
            check_delta = 'delta_5' in st.session_state
            if check_delta:
                delta_5 = monthly_charge - st.session_state['delta_5']

            st.metric(label="Monthly charge", help="median",
                      value=f"${precision(monthly_charge)}", delta=precision(delta_5) if check_delta else None)
            st.session_state['delta_5'] = monthly_charge

        with col6:
            tenure = int(kpi_metrics.get('tenure'))
            check_delta = 'delta_6' in st.session_state
            if check_delta:
                delta_6 = tenure - st.session_state['delta_6']

            st.metric(label="Tenure", help="months (median)",
                      value=tenure, delta=delta_6 if check_delta else None)
            st.session_state['delta_6'] = tenure

        style_metric_cards()

    def show_kpi_plots(df_filtered, target):
        def show_fig(fig):
            # Update the color continuous scale in the existing figure
            return st.plotly_chart(fig)

        def show_figs_in_stcol(figs):
            for i, fig in enumerate(figs):
                if i % 2 == 0:
                    col_a, col_b = st.columns(2)
                    col = col_a
                else:
                    col = col_b

                with col:
                    show_fig(fig)

        def churn_stayed():
            class_counts = df_filtered[target].value_counts().reset_index()
            class_counts.columns = ['churn_class', 'count']
            class_ratio = class_counts.copy()
            class_ratio['ratio'] = class_ratio['count'].apply(
                lambda x: x*100/class_counts['count'].sum())
            class_ratio.drop(columns='count', inplace=True)
            # Visualizing the class distribution of the target variable
            fig = px.pie(
                class_ratio, values='ratio', hole=0.5,
                names='churn_class', title='Churn percentage',
                labels={'churn_class': 'Churn',
                        'ratio': 'Churn rate'},
                hover_data={'ratio': ':.2f'}
            )

            fig = fig.update_layout(
                legend_title_text='Churn',
                legend=dict(
                    x=1,
                    y=1
                ),
                # Reverses the default trace order.
                legend_traceorder='reversed',
            )
            return fig

        def median_tenure_churned_stayed():
            mask = df_filtered['churn'] == 'Yes'

            churned_customers = df_filtered[mask]
            stayed_customers = df_filtered[~mask]

            med_tenure_churned = churned_customers['tenure'].median()
            med_tenure_stayed = stayed_customers['tenure'].median()

            churn = ['No', 'Yes']
            med_tenure = [med_tenure_stayed, med_tenure_churned]

            # Creating the bar plot
            fig = px.bar(
                x=churn,
                y=med_tenure,
                labels={'x': 'Churn',
                        'y': 'Median Tenure', 'color': 'Churn'},
                title='Tenure of Churned vs Stayed Customers',
                color=churn,
                category_orders={'x': churn[::-1]},
            )
            fig = fig.update_layout(legend_traceorder='reversed')
            fig = fig.update_traces(
                texttemplate='%{y:.2s}', textposition='inside')

            return fig

        def default_values(yes_no_churn):
            return ([], None, None, 100 if yes_no_churn else 1)

        def churn_col_stack(column, col_text, title):
            unique_churn = sorted(df_filtered['churn'].unique())
            churn_col = df_filtered.groupby(
                [column]+['churn'])['churn'].count()

            churn_col_df = churn_col.unstack().reset_index()
            unique_col_count = len(df_filtered[column].unique())

            fig = go.Figure()

            # Calculate the total for percentage calculation
            yes_no_churn = unique_churn == ['No', 'Yes']
            churn_col_df['Total'] = (
                churn_col_df['No'] + churn_col_df['Yes']) if yes_no_churn else 1

            x, y, customdata, pct = default_values(yes_no_churn)
            if 'No' in unique_churn:
                churn_col_df.sort_values(
                    by='No', ascending=False, inplace=True)
                churn_col_df['No_pct'] = (
                    churn_col_df['No'] / churn_col_df['Total']) * pct
                x = churn_col_df[column]
                y = churn_col_df['No']
                customdata = churn_col_df['No_pct'].map(
                    lambda x: f'{x:.2f}%') if yes_no_churn else ["Churn"]*unique_col_count

            # Add No trace
            fig = fig.add_trace(go.Bar(
                x=x,
                y=y,
                name='No',
                customdata=customdata,
                texttemplate='%{y}',
                textposition='inside',
                hovertemplate='%{customdata}',
                legendgroup='churn'
            ))

            x, y, customdata, pct = default_values(yes_no_churn)
            if 'Yes' in unique_churn:
                churn_col_df.sort_values(
                    by='Yes', ascending=False, inplace=True)
                churn_col_df['Yes_pct'] = (
                    churn_col_df['Yes'] / churn_col_df['Total']) * pct
                x = churn_col_df[column]
                y = churn_col_df['Yes']
                customdata = churn_col_df['Yes_pct'].map(
                    lambda x: f'{x:.2f}%') if yes_no_churn else ["Churn"]*unique_col_count
            # Add Yes trace
            fig = fig.add_trace(go.Bar(
                x=x,
                y=y,
                name='Yes',
                customdata=customdata,
                texttemplate='%{y}',
                textposition='inside',
                hovertemplate='%{customdata}',
                legendgroup='churn'
            ))

            fig = fig.update_layout(
                barmode='stack',
                title=title,
                xaxis_title=col_text,
                yaxis_title='Number of Customers',
                legend_title='Churn',
            )

            return fig

        def churn_payment_method():
            col = 'payment_method'
            col_text = 'Payment method'
            title = 'Common Payment Methods Among Customers by Churn'
            fig = churn_col_stack(col, col_text, title)

            return fig

        def churn_demography():

            for col in ['gender', 'senior_citizen']:
                col_text = col.title().replace('_', ' ')
                title = f'Churn Distribution by {col_text}'
                fig = churn_col_stack(col, col_text, title)

                yield fig

            def churn_rate_partner():
                def segment_yes(segment):
                    return segment['Yes'] if sorted(df_filtered['churn'].unique()) == ['No', 'Yes'] and 'Yes' in segment.index else 0

                partner_churn_rate = df_filtered[df_filtered['partner'] ==
                                                 'Yes']['churn'].value_counts(normalize=True)
                partner_churn_rate = segment_yes(partner_churn_rate)

                no_partner_churn_rate = df_filtered[df_filtered['partner'] == 'No']['churn'].value_counts(
                    normalize=True)
                no_partner_churn_rate = segment_yes(no_partner_churn_rate)

                dependent_churn_rate = df_filtered[df_filtered['dependents'] == 'Yes']['churn'].value_counts(
                    normalize=True)
                dependent_churn_rate = segment_yes(dependent_churn_rate)

                no_dependent_churn_rate = df_filtered[df_filtered['dependents'] == 'No']['churn'].value_counts(
                    normalize=True)
                no_dependent_churn_rate = segment_yes(no_dependent_churn_rate)

                churn_rates = [partner_churn_rate, no_partner_churn_rate,
                               dependent_churn_rate, no_dependent_churn_rate]

                churn_rates = [round(r*100, 2) for r in churn_rates]

                segments = ['With Partner', 'Without Partner',
                            'With Dependents', 'Without Dependents']

                fig = px.bar(
                    x=segments,
                    y=churn_rates,
                    text_auto=True
                )

                fig = fig.update_layout(
                    title='Churn Rate Based on Partners and Dependents',
                    xaxis_title='Customer Segment',
                    yaxis_title='Churn Rate (%)',
                )

                # Set y-axis limits from 0 to 100
                fig = fig.update_yaxes(range=[0, 100])

                return fig

            yield churn_rate_partner()

        def churn_contract():
            churn_contract = df_filtered.groupby(
                'contract')['churn'].value_counts().reset_index()

            fig = px.bar(churn_contract, x='contract', y='count',
                         color='churn', barmode='group')

            fig = fig.update_layout(
                title='Churn Distribution by Contract Term',
                xaxis_title='Contract Term',
                yaxis_title='Count',
                legend_title='Churn',
                legend_traceorder='reversed',
            )

            fig = fig.update_traces(
                texttemplate='%{y}', textposition='outside')

            return fig

        def churn_internet_service():
            col = 'internet_service'
            col_text = 'Internet service'
            title = 'Churn by Internet Service Availability'
            fig = churn_col_stack(col, col_text, title)

            return fig

        def churn_online_services():
            online_related_services = ['online_security', 'online_backup',
                                       'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']

            for col in online_related_services:
                col_text = col.title().replace('_', ' ')
                title = f'Churn Distribution by {col_text}'
                fig = churn_col_stack(col, col_text, title)

                yield fig

        def trend_by_tenure():
            unique_churn = sorted(df_filtered['churn'].unique())
            med_monthly_charges = df_filtered.groupby(
                'tenure')['monthly_charges'].median().reset_index()

            # Plotly line chart
            fig = px.line(med_monthly_charges, x='tenure', y='monthly_charges',
                          title='Trend of Monthly Charges by Tenure')
            fig = fig.update_layout(
                xaxis_title='Tenure',
                yaxis_title='Monthly Charges',
            )
            yield fig

            # Calculate churn rate by tenure
            yes_no_churn = unique_churn == ['No', 'Yes']

            churn_tenure = df_filtered.groupby(
                'tenure')['churn'].value_counts().unstack(fill_value=0)
            churn_tenure['churn_rate'] = (churn_tenure['Yes'] /
                                          churn_tenure.sum(axis=1) * 100) if yes_no_churn else 1
            churn_tenure = churn_tenure.reset_index()

            # Plotly line chart
            fig = px.line(churn_tenure, x='tenure', y='churn_rate',
                          title='Trend of Churn Rate by Tenure')
            fig = fig.update_layout(
                xaxis_title='Tenure',
                yaxis_title='Churn Rate (%)',
            )
            yield fig

        # Show the visualizations
        col1, col2 = st.columns(2)

        with col1:
            show_fig(churn_stayed())

        with col2:
            show_fig(median_tenure_churned_stayed())

        col3, col4 = st.columns(2)

        with col3:
            show_fig(churn_contract())

        with col4:
            show_fig(churn_payment_method())

        st.header("Churn Distribution by Demography")
        show_figs_in_stcol(churn_demography())

        st.header("Churn by Internet Service Availabilty")
        show_fig(churn_internet_service())

        st.header("Churn Distribution by Service Usage")
        show_figs_in_stcol(churn_online_services())

        st.header("Trend by Tenure")
        show_figs_in_stcol(trend_by_tenure())

    """ 
        KPI view
        Show side bar and get the selected options as boolean mask of the dataframe
    """
    kpi_container = st.container()

    with kpi_container:
        # First containt to paint
        kpi_container.header("Key Performance Indicators 🎯")

        categoricals = df.select_dtypes(exclude=['number']).columns.tolist()

        with st.sidebar:
            mask = kpi_sidebar(df, categoricals)

        # Filter and Filtered df
        df_filtered = df.copy()
        df_filtered = filter_data(df_filtered, mask)

        # Calculate KPIs
        kpis = calculate_kpis(df_filtered)
        kpi_names = ['no_of_customers', 'churned_customers',
                     'churn_rate', 'total_revenue', 'monthly_charge', 'tenure']

        show_kpi_metrics(kpis, kpi_names)

        show_kpi_plots(df_filtered, target)
