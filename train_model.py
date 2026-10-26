"""Train and save an Iris classifier used by the inference API."""
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump({"model": model, "target_names": iris.target_names.tolist(), "feature_names": iris.feature_names}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}; held-out accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()
