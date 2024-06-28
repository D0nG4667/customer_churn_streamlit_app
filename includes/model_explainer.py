import os

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

from includes.markdown import log_reg_insight
from includes.func_in_pipeline import target, numerical_features, categorical_features_new
from config.settings import ENCODER_FILE, MODELS


def model_explainer(df):
    """ All Functions """
    # Load pipelines
    @st.cache_resource(show_spinner="Loading pipelines...")
    def load_all_pipelines():
        def load_pipeline(model_name):
            return joblib.load(os.path.join(MODELS, model_name))
        # List all pipelines in the models directory
        all_models = [f for f in os.listdir(
            MODELS) if os.path.isfile(os.path.join(MODELS, f))]
        all_pipelines = {m.split(".")[0]: load_pipeline(m) for m in all_models}
        return all_pipelines

    # Load encoder
    @st.cache_resource(show_spinner="Loading encoder...")
    def load_encoder():
        return joblib.load(ENCODER_FILE)

    @st.cache_data(show_spinner=False)
    def train_test_split_encode(df):
        df.dropna(subset=target, inplace=True)
        # Split the data into X and y
        X = df.drop(columns=[target])
        y = df[[target]]
        # Split the X and y into train and eval
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, train_size=0.8, random_state=2024, stratify=y)

        # Ensure y_eval is 1D using [target]
        y_eval_encoded = encoder.transform(y_eval[target])

        return X_train, X_eval, y_train, y_eval, y_eval_encoded

    def feature_importances(pipeline):
        # Get the numerical feature names after transformation
        numerical_features_transformed = pipeline.named_steps['preprocessor'].named_transformers_[
            'num_pipeline'].named_steps['scaler'].get_feature_names_out(numerical_features)
        categorical_features_transformed = pipeline.named_steps['preprocessor'].named_transformers_[
            'cat_pipeline'].named_steps['encoder'].get_feature_names_out(categorical_features_new)

        # Get the feature names after transformation
        feature_columns = np.concatenate(
            (numerical_features_transformed, categorical_features_transformed))

        # Access the coefficients since best model is logistic regression
        coefficients = pipeline.named_steps['classifier'].coef_[0]

        coefficients_df = pd.DataFrame(
            {'Feature': feature_columns, 'Coefficient': coefficients})

        # Magnitude of impact
        coefficients_df['Absolute Coefficient'] = np.abs(
            coefficients_df['Coefficient'])
        coefficients_df.sort_values(
            by="Absolute Coefficient", ascending=True, inplace=True)

        fig = px.bar(
            coefficients_df,
            x='Coefficient',
            y='Feature',
            orientation='h',  # Set orientation to horizontal
            title='Feature Importances - Logistic Regression Coefficients',
            labels={'Coefficient': 'Coefficient Value', 'Feature': 'Features'},
            height=700,
            color='Coefficient'
        )

        return fig

    def all_confusion_matrix():
        _, X_eval, _, y_eval, y_eval_encoded = train_test_split_encode(df)

        target_class = y_eval[target].unique().tolist()

        for model_name, pipeline in all_pipelines.items():
            # Predict and calculate performance scores
            y_pred = pipeline.predict(X_eval)

            # Defining the Confusion Matrix
            model_conf_mat = confusion_matrix(y_eval_encoded, y_pred)

            # Use Plotly Express to create the confusion matrix heatmap
            fig = fig = px.imshow(
                model_conf_mat,
                labels=dict(x='Predicted', y='Actual', color='Count'),
                x=target_class,  # Prediction labels
                y=target_class,  # Actual labels
                text_auto=True,  # Automatically add text in each cell
                color_continuous_scale='RdBu',  # Color scale
                width=700,
                height=700
            )

            # Add title and adjust layout
            fig = fig.update_layout(
                title=f'Confusion Matrix- {model_name}',
                # Adjust ticks to match number of classes
                xaxis_nticks=len(model_conf_mat),
                yaxis_nticks=len(model_conf_mat),
            )

            yield fig

    @st.cache_data(show_spinner=False)
    def roc_auc_curve(_pipeline, pipeline_name):
        _, X_eval, _, _, y_eval_encoded = train_test_split_encode(df)
        y_score = _pipeline.predict_proba(X_eval)[:, 1]
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(
            y_eval_encoded, y_score)
        roc_auc = auc(fpr, tpr)
        fig = px.area(
            x=fpr,
            y=tpr,
            title=f'ROC Curve (AUC={roc_auc:.2f}) - {pipeline_name}',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=800,
            height=800
        )
        fig = fig.add_shape(
            type='line',
            line=dict(dash='dash'),
            x0=0,
            x1=1,
            y0=0,
            y1=1
        )

        fig = fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig = fig.update_xaxes(constrain='domain')

        return fig

    @st.cache_data(show_spinner=False)
    def all_roc_curves(_all_pipelines):
        _, X_eval, _, _, y_eval_encoded = train_test_split_encode(df)

        fig = go.Figure()

        for model_name, pipeline in _all_pipelines.items():
            y_score = pipeline.predict_proba(X_eval)[:, 1]

            fpr, tpr, _ = roc_curve(y_eval_encoded, y_score)

            roc_auc = auc(fpr, tpr)

            fig = fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{
                                model_name} (AUC={roc_auc:.2f})'))

            fig = fig.update_layout(
                title=f'ROC AUC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(
                    x=1.02,
                    y=0.98
                ),
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=1024,
                height=800
            )

        fig = fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        return fig

    def show_fig(fig, col=None):
        # Update the color continuous scale in the existing figure
        if col is not None:
            with col:
                st.plotly_chart(fig)
        else:
            st.plotly_chart(fig, use_container_width=True)

    def show_figs_in_stcol(figs, one_col=False):
        for i, fig in enumerate(figs):
            if i % 2 == 0 and not one_col:
                col_a, col_b = st.columns(2)
                col = col_a
            elif one_col:
                col = None
            else:
                col = col_b

            show_fig(fig, col)

    """
        Show the Model Explainer view and visualizations        
    """
    explainer_container = st.container()
    fig_container = st.container()

    with explainer_container:
        # First containt to paint
        st.header("Model explainer 💡")
        st.sidebar.title("Dashboard filters...")

        toggles = st.session_state.get('toggles', [False]*3)

        def handle_toggle(toggle):
            toggles = [False]*3
            toggles[toggle] = True
            st.session_state['toggles'] = toggles
            return toggles


        all_cf_toggle = st.sidebar.toggle(
            "All confusion matrix", toggles[0], on_change=handle_toggle, args=[0], key='all_cf_toggle')
        all_roc_toggle = st.sidebar.toggle(
            "All AUC ROC Curves", toggles[1], on_change=handle_toggle, args=[1], key='all_roc_toggle')

        encoder = load_encoder()
        all_pipelines = load_all_pipelines()
        model_names = list(all_pipelines.keys())

        selected_pipeline_name = st.sidebar.selectbox(
            "Select a model",
            options=model_names,
            index=5,
            placeholder="Choose a model...",
            key='selected_pipeline'
        )
        selected_pipeline = all_pipelines.get(selected_pipeline_name)

        st.subheader(f"{selected_pipeline_name} pipeline preview")

        with st.expander("Expand to peek the pipeline", icon="👀"):
            st.code(selected_pipeline, line_numbers=True)

        if selected_pipeline_name == "LogisticRegression":
            st.toast(f"{selected_pipeline_name} is the best model", icon="🎊")
            if "balloons" not in st.session_state:
                st.balloons()
                st.session_state["balloons"] = True

            st.sidebar.toggle(
                "Feature importances", toggles[2], on_change=handle_toggle, args=[2], key='feat_imp_toggle')

            if st.session_state['feat_imp_toggle']:
                with fig_container.empty().container():
                    st.header("Feature importances")
                    show_fig(feature_importances(selected_pipeline))
                    with st.expander("Expand to view the feature importances insights", icon="💡"):
                        st.markdown(log_reg_insight)

        all_conf_mat_dic = {m: cf for m, cf in zip(
            model_names, all_confusion_matrix())}

        if all_cf_toggle:
            with fig_container.empty().container():
                st.header("All Confusion matrix")
                show_figs_in_stcol(all_confusion_matrix())
        elif all_roc_toggle:
            with fig_container.empty().container():
                st.header("AUC ROC Curves")
                show_fig(all_roc_curves(all_pipelines))
        else:
            with fig_container:
                selected_mat_fig = all_conf_mat_dic.get(selected_pipeline_name)
                st.header("Confusion matrix")
                show_fig(selected_mat_fig)

                st.header("AUC ROC Curve")
                selected_roc_curve = roc_auc_curve(
                    selected_pipeline, selected_pipeline_name)
                show_fig(selected_roc_curve)
