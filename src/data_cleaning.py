import pandas as pd
import os

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

def save_processed_splits(X_train, X_test, y_train, y_test, folder="../data/processed"):
    os.makedirs(folder, exist_ok=True)
    X_train.to_csv(f"{folder}/X_train.csv", index=False)
    X_test.to_csv(f"{folder}/X_test.csv", index=False)
    y_train.to_csv(f"{folder}/y_train.csv", index=False)
    y_test.to_csv(f"{folder}/y_test.csv", index=False)


def load_processed_splits(folder="../data/processed"):
    X_train = pd.read_csv(f"{folder}/X_train.csv")
    X_test = pd.read_csv(f"{folder}/X_test.csv")
    y_train = pd.read_csv(f"{folder}/y_train.csv").squeeze("columns")
    y_test = pd.read_csv(f"{folder}/y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test