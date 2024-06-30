import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

from includes.markdown import num_uni_biv_insight, num_mul_violin_insight, num_mul_pca_insight, cat_uni_biv_insight, cat_mul_insight


def eda(df, target):
    """ All Functions """
    def show_numericals(eda_type):
        st.subheader('Numerical features 🔢')

        numericals = df.select_dtypes(include=['number']).columns.tolist()

        def show_univariate():
            st.subheader("Univariate Analysis")

            # Visualize their distributions
            for column in df[numericals].columns:
                fig1 = px.violin(df, x=column, box=True)

                fig2 = px.histogram(df, x=column)

                # Create a subplot layout with 1 row and 2 columns
                fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Violin plot of the {column} column",
                                                                    f"Distribution of the {column} column"))

                # Add traces from fig1 to the subplot
                for trace in fig1.data:
                    fig = fig.add_trace(trace, row=1, col=1)

                # Add traces from fig2 to the subplot
                for trace in fig2.data:
                    fig = fig.add_trace(trace, row=1, col=2)

                # Update layout
                fig = fig.update_layout(title_text=f"Exploring the {column} feature",
                                        showlegend=True,
                                        legend_title_text=target
                                        )
                yield fig

        def show_bivariate():
            st.subheader("Bivariate Analysis")
            for column in numericals:
                # Visualizing the distribution of the numericals in the columns by churn
                fig = px.violin(
                    df,
                    x=target,
                    y=column,
                    color=target,
                    box=True,
                    title=f"Distribution of users in the {
                        column} column by churn"
                )

                yield fig

            # Calculate correlation matrix
            numeric_correlation_matrix = df[numericals].corr()
            # Create heatmap trace
            heatmap_trace = go.Heatmap(
                z=numeric_correlation_matrix.values,
                x=numeric_correlation_matrix.columns,
                y=numeric_correlation_matrix.index,
                colorbar=dict(title='Correlation coefficient'),
                texttemplate='%{z:.3f}',
            )
            # Create figure
            fig = go.Figure(data=[heatmap_trace])
            # Update layout
            fig = fig.update_layout(
                title='Correlation Matrix Heatmap (Numeric Features)',
            )

            yield fig

        @st.cache_data(show_spinner=False)
        def show_multivariate(df):
            st.subheader("Multivariate Analysis")

            def violin():
                fig = go.Figure()

                fig = fig.add_trace(
                    go.Violin(
                        x=df['payment_method'][df['churn'] == 'No'],
                        y=df['tenure'][df['churn'] == 'No'],
                        legendgroup='No', scalegroup='No', name='No',
                        side='positive'
                    )
                )

                fig = fig.add_trace(
                    go.Violin(
                        x=df['payment_method'][df['churn'] == 'Yes'],
                        y=df['tenure'][df['churn'] == 'Yes'],
                        legendgroup='Yes', scalegroup='Yes', name='Yes',
                        side='negative'
                    )
                )

                fig = fig.update_traces(meanline_visible=True)
                fig = fig.update_layout(
                    xaxis_title='Payment Method',
                    yaxis_title='Tenure',
                    violingap=0,
                    violinmode='overlay'
                )

                return fig

            def pca():
                pca = PCA(n_components=2)

                X = df[numericals+[target]].dropna()

                components = pca.fit_transform(X.drop(columns=target))

                total_var = pca.explained_variance_ratio_.sum() * 100

                fig = px.scatter(
                    components, x=0, y=1, color=X['churn'],
                    title=f'Total Explained Variance: {total_var:.2f}%',
                    labels={'0': 'PC 1', '1': 'PC 2'}
                )

                return fig

            return violin(), pca()

        # Display the numericals
        if "Univariate" in eda_type:
            for fig in show_univariate():
                st.plotly_chart(fig)

        if "Bivariate" in eda_type:
            for fig in show_bivariate():
                st.plotly_chart(fig)

            # Display explander to show Key insights
            with st.expander("Expand to view the key insights", icon="💡"):
                st.markdown(num_uni_biv_insight)

        if "Multivariate" in eda_type:
            mul_violin, mul_pca = show_multivariate(df)
            st.plotly_chart(mul_violin)
            with st.expander("Expand to view the key insights", icon="💡"):
                st.markdown(num_mul_violin_insight)

            st.plotly_chart(mul_pca)
            with st.expander("Expand to view the key insights", icon="💡"):
                st.markdown(num_mul_pca_insight)

    def show_categoricals(eda_type):
        st.subheader('Categorical features 🔡')

        categoricals = df.select_dtypes(
            include=['object', 'category']).columns.tolist()

        if "customer_id" in categoricals:
            categoricals.remove("customer_id")

            # Visualizing the distribution of the columns with categorical values and with respect to churn
        def show_uni_biv():
            st.subheader("Univariate & Bivariate Analysis")

            for column in categoricals:
                if column != target:  # Exclude the 'churn' column
                    # Visualizing the distribution of the categories in the columns
                    fig1 = px.histogram(df, x=column, text_auto=True, opacity=0.5,
                                        title=f"Distribution of users in the {column} column")

                    # Visualizing the distribution of the categories in the columns by churn
                    fig2 = px.histogram(df, x=column, color=target, text_auto=".1f",
                                        title=f"Distribution of users in the {column} column by churn")

                    # Create a subplot layout with 1 row and 2 columns
                    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Distribution of users in the {column}",
                                                                        f"Distribution by churn in the {column}"))

                    # Add traces from fig1 to the subplot
                    for trace in fig1.data:
                        fig = fig.add_trace(trace, row=1, col=1)

                    # Add traces from fig2 to the subplot
                    for trace in fig2.data:
                        fig = fig.add_trace(trace, row=1, col=2)

                    # Update layout
                    fig = fig.update_layout(
                        title_text=f"Univariate vs Bivariate Distributions- {
                            column} feature",
                        showlegend=True,
                        legend_title_text=target
                    )

                    yield fig
                else:
                    # Visualizing the distribution of the target variable
                    fig = px.histogram(df, x=column, text_auto=True, color=column,
                                       title=f"Distribution of users in the {column} column")
                    yield fig

        @st.cache_data(show_spinner=False)
        def show_multivariate(df):
            st.subheader("Multivariate Analysis")

            df_train_categoricals = df[categoricals].dropna()
            label_encoder = LabelEncoder()
            df_train_cat_viz = df_train_categoricals.apply(
                label_encoder.fit_transform)
            _, p_values = chi2(df_train_cat_viz.drop(
                target, axis=1), df_train_cat_viz[target])
            chi2_results = pd.DataFrame(p_values, index=df_train_categoricals.drop(
                target, axis=1).columns, columns=[target])
            chi2_results = chi2_results.sort_values(by=target, ascending=False)
            chi2_results = chi2_results.sort_values(by=target, ascending=True)
            data = go.Heatmap(
                z=chi2_results.values,
                x=chi2_results.columns,
                y=chi2_results.index+' -',
                colorbar=dict(title='P-value'),
                hovertemplate='%{y} %{x}: p=%{z}',
                texttemplate='%{z}'
            )
            fig = go.Figure(data)

            fig = fig.update_layout(
                title='Chisquare association between Categorical Variables and Churn',
                width=900,
                height=600
            )

            return fig

        # Display the categoricals
        if set(['Univariate', 'Bivariate', 'Univariate & Bivariate']).intersection(eda_type):
            for fig in show_uni_biv():
                st.plotly_chart(fig)

            with st.expander("Expand to view the key insights", icon="💡"):
                st.markdown(cat_uni_biv_insight)

        if "Multivariate" in eda_type:
            fig = show_multivariate(df)
            st.plotly_chart(fig)

            with st.expander("Expand to view the key insights", icon="💡"):
                st.markdown(cat_mul_insight)

    """
        Show the Model Explainer view and visualizations
    """
    eda_container = st.container()

    with eda_container:
        # First containt to paint
        st.header("Exploratory Data Analysis 🔎")

        with st.sidebar:
            st.sidebar.title("Dashboard filters...")

            num_toggle = st.sidebar.toggle(
                "Numerical features", True, key='numericals')

            cat_toggle = st.sidebar.toggle(
                "Categorical features", True, key='categoricals')

            eda_type_options = ['Univariate', 'Bivariate', 'Multivariate'] if num_toggle else [
                'Univariate & Bivariate', 'Multivariate']

            eda_type = st.sidebar.multiselect(
                "Select the EDA focus", eda_type_options, default=eda_type_options, key="eda_type")

        # Handle display according to numerical and categoricals toggles
        if num_toggle:
            show_numericals(eda_type)

        if cat_toggle:
            show_categoricals(eda_type)
