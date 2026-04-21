import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


DATA_PATH = Path("Tweets (2).csv")
TARGET = "airline_sentiment"

# Mapping explicite pour encoder la cible.
SENTIMENT_TO_INT = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def basic_report(df: pd.DataFrame) -> None:
    """Affiche les informations de base du dataset."""
    print("=== Apercu ===")
    print(df.head())
    print("\n=== Shape (observations, variables) ===")
    print(df.shape)
    print("\n=== Types des variables ===")
    print(df.dtypes)
    print("\n=== Valeurs manquantes ===")
    print(df.isna().sum())
    print("\n=== Distribution de la cible ===")
    print(df[TARGET].value_counts(dropna=False))


def drop_useless_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les variables non utiles pour une classification basee sur le texte du tweet.

    On conserve uniquement la variable textuelle principale `text`.
    """
    columns_to_keep = ["text"]
    return df.loc[:, columns_to_keep].copy()


def clean_tweet_text(text: str) -> str:
    """Nettoyage maison d'un tweet."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"&amp;", "and", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Applique la fonction de nettoyage sur la colonne text."""
    processed = df.copy()
    processed["text"] = processed["text"].apply(clean_tweet_text)
    return processed


def main() -> None:
    # 1) Chargement du dataset
    df = pd.read_csv(DATA_PATH)

    # 2) Informations de base
    basic_report(df)

    # 3) Variables utiles pour la classification de airline_sentiment
    useful_variables = [
        "text",
        "airline",
        "airline_sentiment_confidence",
        "retweet_count",
        "user_timezone",
        "tweet_location",
        "tweet_created",
    ]
    useful_variables = [col for col in useful_variables if col in df.columns]
    print("\n=== Variables potentiellement utiles ===")
    print(useful_variables)

    # 4) Encodage de la cible en {0, 1, 2}
    df[TARGET] = df[TARGET].map(SENTIMENT_TO_INT)
    if df[TARGET].isna().any():
        raise ValueError("Certaines modalites de airline_sentiment ne sont pas mappees.")
    df[TARGET] = df[TARGET].astype(int)

    # 5) Separation train+validation / test
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Optionnel: split train / validation interne (a partir de train_val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=42,
        stratify=y_train_val,
    )

    print("\n=== Tailles des jeux ===")
    print(f"Train+Val: {X_train_val.shape}, Test: {X_test.shape}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 6) Pipeline avec:
    #    - suppression des variables inutiles
    #    - fonction maison de pretraitement des tweets
    preprocessing_pipeline = Pipeline(
        steps=[
            ("drop_useless", FunctionTransformer(drop_useless_variables, validate=False)),
            ("tweet_preprocess", FunctionTransformer(preprocess_tweets, validate=False)),
        ]
    )

    X_train_ready = preprocessing_pipeline.fit_transform(X_train)
    X_val_ready = preprocessing_pipeline.transform(X_val)
    X_test_ready = preprocessing_pipeline.transform(X_test)

    print("\n=== Exemple de tweets pretraites (train) ===")
    print(X_train_ready.head())

    print("\nPipeline cree et applique avec succes.")


if __name__ == "__main__":
    main()
