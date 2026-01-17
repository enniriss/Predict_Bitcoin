import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# Chemins absolus
RAW_DATA_PATH = "c:/Users/Utilisateur/Documents/Cour/Data Science/GROUPE-EXAM-FINAL/Predict_Bitcoin/src/Data/marketing_campaign.csv"
PROCESSED_DATA_PATH = "c:/Users/Utilisateur/Documents/Cour/Data Science/GROUPE-EXAM-FINAL/Predict_Bitcoin/src/Data/processed/marketing_campaign_processed.csv"
MODEL_PATH = "c:/Users/Utilisateur/Documents/Cour/Data Science/GROUPE-EXAM-FINAL/Predict_Bitcoin/src/models/ecommerce_model.pkl"
KMEANS_MODEL_PATH = "c:/Users/Utilisateur/Documents/Cour/Data Science/GROUPE-EXAM-FINAL/Predict_Bitcoin/src/models/kmeans_model.pkl"

USELESS_COLUMNS = ["ID", "Dt_Customer", "Year_Birth"]
CAT_FEATURES = ["Education", "Marital_Status"]
TARGET = "Response"
BEST_K = 2  


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def drop_constant_and_useless(df: pd.DataFrame) -> pd.DataFrame:
    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    to_drop = list(set(constant_cols + USELESS_COLUMNS))
    return df.drop(columns=to_drop, errors="ignore")


def iqr_mask(s: pd.Series, factor: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - factor * iqr, q3 + factor * iqr
    return (s < low) | (s > high)


def cap_outliers_with_mean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        m = df[col].mean()
        mask = iqr_mask(df[col])
        df.loc[mask, col] = m
    return df


def merge_purchases(df: pd.DataFrame) -> pd.DataFrame:
    purchase_cols = [c for c in df.columns if "Purchase" in c]
    if "NumDealsPurchases" in purchase_cols:
        purchase_cols.remove("NumDealsPurchases")
    df["NumPurchases"] = df[purchase_cols].sum(axis=1)
    return df.drop(columns=purchase_cols, errors="ignore")


def preprocess_split(df: pd.DataFrame):
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")),
         ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        [("num", numeric_pipe, num_features),
         ("cat", cat_pipe, CAT_FEATURES)],
        remainder="drop",
    )
    return X_train, X_test, y_train, y_test, preprocessor


def train_model(preprocessor, X_train, y_train):
    model = LogisticRegression(max_iter=2000, random_state=42)
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✅ Saved: {path}")


def save_processed(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Processed data saved to {path}")


def add_clusters(df: pd.DataFrame, preprocessor, X: pd.DataFrame, k: int = BEST_K):
    X_transformed = preprocessor.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_transformed)
    df = df.copy()
    df["Cluster"] = clusters
    return df, kmeans


def main():
    df = load_data(RAW_DATA_PATH)
    df = drop_constant_and_useless(df)

    mnt_cols = [c for c in df.columns if "Mnt" in c]
    purchase_cols = [c for c in df.columns if "Purchase" in c and c != "NumDealsPurchases"]
    df = cap_outliers_with_mean(df, mnt_cols + purchase_cols)

    df = merge_purchases(df)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_split(df)
    pipe = train_model(preprocessor, X_train, y_train)
    evaluate(pipe, X_test, y_test)
    save_pickle(pipe, MODEL_PATH)

    X_full = df.drop(columns=[TARGET])
    df_with_clusters, kmeans_model = add_clusters(df, preprocessor, X_full, k=BEST_K)
    save_pickle(kmeans_model, KMEANS_MODEL_PATH)
    save_processed(df_with_clusters, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()