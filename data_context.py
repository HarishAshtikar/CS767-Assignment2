"""Dataset perception helpers."""

import io

import pandas as pd


def build_data_context(df: pd.DataFrame) -> str:
    """Build the compact dataset summary sent to the model."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()

    return f"""
Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns
Columns: {list(df.columns)}

Dataframe info:
{df_info}

Dtypes:
{df.dtypes.to_string()}

First 5 rows:
{df.head().to_string()}

Basic statistics:
{df.describe(include="all").to_string()}

Missing values:
{df.isnull().sum().to_string()}
"""
