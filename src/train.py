import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_data():
    """Load the Iris dataset and split it into training and testing data."""

    iris = load_iris()

    return train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )


def train_model(X_train, y_train, n_estimators=100, max_depth=None):
    """Train the Random Forest classification model."""

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(y_test, predictions)

    return accuracy, report


def main():

    # Load and split the Iris dataset
    X_train, X_test, y_train, y_test = load_data()

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Display evaluation results
    print("\n===== Model Evaluation Results =====")

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\n===== Classification Report =====")
    print(report)

    # Save the trained model
    model_path = "models/iris_model.joblib"

    joblib.dump(model, model_path)

    print(f"\nModel saved successfully to: {model_path}")


if __name__ == "__main__":
    main()