import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    iris = load_iris()

    return train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )


def train_model(X_train, y_train, n_estimators=100, max_depth=None):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(y_test, predictions)

    return accuracy, report


def main():

    # Load and split data
    X_train, X_test, y_train, y_test = load_data()

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(report)

    # Save trained model
    joblib.dump(model, "models/iris_model.joblib")

    print("\nModel saved to models/iris_model.joblib")


if __name__ == "__main__":
    main()