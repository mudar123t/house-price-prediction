import pandas as pd


def load_raw_data(path="../data/raw/Raw-Data.xlsx"):
    """
    Load the raw house dataset from Excel.
    """
    return pd.read_excel(path)


def drop_empty_rows(df):
    """
    Drop fully empty template rows (rows with no data in any column).
    """
    return df.dropna(how="all")


def clean_data(df):
    """
    Full cleaning pipeline.
    Currently the source file is well-maintained (no missing values),
    but this pipeline guards against future dirty entries as more
    houses get added:
      - drops fully empty template rows
      - drops rows missing price (can't train without the target)
    """
    df = drop_empty_rows(df)
    df = df.dropna(subset=["السعر"])
    return df