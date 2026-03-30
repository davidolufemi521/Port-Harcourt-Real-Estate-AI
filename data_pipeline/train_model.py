import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "RIVERS_READY_FOR_ML.csv"
MODEL_PATH = "rivers_rent_model.pkl"
FEATURES_PATH = "model_features.pkl"

DROP_BEFORE_MODELING = ["source", "url", "title", "period"]
DEDUP_EXCLUDE = ["source", "url", "title"]
NUMERIC_FEATURES = ["bedrooms", "bathrooms", "toilets", "size_sqm"]


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dedup_subset = [col for col in df.columns if col not in DEDUP_EXCLUDE]

    before_rows = len(df)
    df = df.drop_duplicates(subset=dedup_subset, keep="first").copy()
    removed_rows = before_rows - len(df)

    print(f"Loaded {before_rows} rows from {path}")
    print(f"Removed {removed_rows} duplicate listings based on non-scrape columns")

    return df


def build_features(df: pd.DataFrame):
    feature_cols = [col for col in df.columns if col not in DROP_BEFORE_MODELING + ["price"]]
    categorical_features = [col for col in feature_cols if col not in NUMERIC_FEATURES]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(random_state=42)),
        ]
    )

    return df[feature_cols], df["price"], model, feature_cols


def evaluate_model(X: pd.DataFrame, y: pd.Series, model: Pipeline) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    holdout_mae = mean_absolute_error(y_test, predictions)
    holdout_r2 = r2_score(y_test, predictions)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
    )

    cv_mae = -scores["test_mae"].mean()
    cv_r2 = scores["test_r2"].mean()

    print("\nEvaluation on cleaned raw dataset")
    print(f"Holdout MAE: {holdout_mae:,.2f}")
    print(f"Holdout R2: {holdout_r2:.4f}")
    print(f"5-fold CV MAE: {cv_mae:,.2f}")
    print(f"5-fold CV R2: {cv_r2:.4f}")


def train_and_save_final_model(X: pd.DataFrame, y: pd.Series, model: Pipeline, feature_cols):
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    print(f"\nSaved trained pipeline to {MODEL_PATH}")
    print(f"Saved expected raw input features to {FEATURES_PATH}")


def main():
    print("Loading and cleaning the Rivers rental dataset...\n")
    df = load_and_clean_data(DATA_PATH)
    X, y, model, feature_cols = build_features(df)
    evaluate_model(X, y, model)
    train_and_save_final_model(X, y, model, feature_cols)
    print("\nTraining complete. The app can now load the cleaned-data regression pipeline.")


if __name__ == "__main__":
    main()
