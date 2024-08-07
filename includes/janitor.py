import pandas as pd
import re


class Janitor:
    def __init__(self):
        pass

    def clean_dataframe(self, df):
        # Apply all cleaning procedure in sequence
        df = df.copy()  # First make a copy to preserve integrity of the old df
        df = self.drop_duplicates(df)
        df = self.snake_case_columns(df)
        df = self.fix_none(df)
        df = self.fix_datatypes(df)
        df = self.clean_categoricals(df)
        df = self.dropna_target(df)
        df = df.reset_index(drop=True) # Fix index
        return df

    def drop_duplicates(self, df):
        return df.drop_duplicates() if df.duplicated().sum() > 0 else df

    def snake_case_columns(self, df):
        pattern = r'(?<!^)(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])'
        df.columns = [re.sub(pattern, '_', column).lower()
                      for column in df.columns]
        return df

    def fix_none(self, df):
        def replace_none(value):
            like_nan = {'none', ''}
            if pd.isnull(value) or (isinstance(value, str) and (value.lower().strip() in like_nan)):
                value = pd.NA
            return value

        return df.map(replace_none)

    def fix_datatypes(self, df):
        col_to_fix = {'total_charges', 'senior_citizen'}
        if col_to_fix.issubset(df.columns):
            df['total_charges'] = pd.to_numeric(
                df['total_charges'], errors='coerce')
            df['senior_citizen'] = df.senior_citizen.astype(str)
        return df

    def clean_with_corrections(self, df: pd.DataFrame, column_names: list, corrections: dict) -> pd.DataFrame:
        corrected_df = df.copy()

        for column_name in column_names:
            for correction, keywords in corrections.items():
                corrected_df[column_name] = corrected_df[column_name].apply(
                    lambda x: correction if (pd.notna(x) and str(x) in keywords) else x)

        return corrected_df

    def clean_categoricals(self, df):
        corrections = {
            "No": ["False", "0", "No phone service", "No internet service"],
            "Yes": ["True", "1"]
        }
        categoricals = df.select_dtypes(
            include=['object', 'category']).columns.tolist()
        return self.clean_with_corrections(df, categoricals, corrections)

    # Drop rows with missing values in target column and reset index
    def dropna_target(self, df):
        return df.dropna(subset='churn') if 'churn' in df.columns else df
