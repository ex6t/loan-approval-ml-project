import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def preprocess_data(df):
    X = df.drop(columns=["loan_id", "loan_status"])
    y = df["loan_status"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        X[col] = X[col].str.strip()

    y = y.str.strip()

    X_encoded = X.copy()
    feature_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        feature_encoders[col] = le

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    return X_encoded, y_encoded, feature_encoders, target_encoder


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

    return accuracy


def save_artifacts(model, target_encoder, feature_encoders):
    joblib.dump(model, "models/random_forest_model.joblib")
    joblib.dump(target_encoder, "models/target_encoder.joblib")
    joblib.dump(feature_encoders, "models/feature_encoders.joblib")

    print("\nSaved model artifacts successfully.")
    print("- models/random_forest_model.joblib")
    print("- models/target_encoder.joblib")
    print("- models/feature_encoders.joblib")


def main():
    print("Loading dataset...")
    df = load_data("data/loan_approval_dataset.csv")

    print("Preprocessing data...")
    X, y, feature_encoders, target_encoder = preprocess_data(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training model...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test, target_encoder)
    save_artifacts(model, target_encoder, feature_encoders)


if __name__ == "__main__":
    main()
