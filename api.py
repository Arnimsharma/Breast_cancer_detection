from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Define the 30 features in correct order
feature_order = [
    "mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness",
    "mean_compactness","mean_concavity","mean_concave_points","mean_symmetry","mean_fractal_dimension",
    "radius_error","texture_error","perimeter_error","area_error","smoothness_error",
    "compactness_error","concavity_error","concave_points_error","symmetry_error","fractal_dimension_error",
    "worst_radius","worst_texture","worst_perimeter","worst_area","worst_smoothness",
    "worst_compactness","worst_concavity","worst_concave_points","worst_symmetry","worst_fractal_dimension"
]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Build input array in correct order
    input_values = []
    for feature in feature_order:
        try:
            input_values.append(float(data[feature]))
        except KeyError:
            input_values.append(0.0)

    input_array = np.array([input_values])

    # Prediction and probabilities
    prediction = model.predict(input_array)[0]          # 0 = malignant, 1 = benign
    proba = model.predict_proba(input_array)[0]         # [P(malignant), P(benign)]

    malignant_prob = float(proba[0] * 100)
    benign_prob = float(proba[1] * 100)

    if prediction == 0:
        label = "Malignant (Cancer Detected)"
    else:
        label = "Benign (No Cancer Detected)"

    return jsonify({
        "status": "success",
        "prediction_label": label,
        "prediction_code": int(prediction),
        "malignant_probability": round(malignant_prob, 2),
        "benign_probability": round(benign_prob, 2)
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Returns model performance metrics for dashboard:
    - accuracy
    - confusion matrix
    - class counts
    - top feature importances
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign
    feature_names = list(data.feature_names)

    # Split same way as training script (you can adjust if needed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # [[TN, FP],[FN, TP]]

    total_samples = int(len(y))
    malignant_count = int(np.sum(y == 0))
    benign_count = int(np.sum(y == 1))

    # Feature importances (RandomForest has this)
    importances = getattr(model, "feature_importances_", None)
    feature_importance_list = []
    if importances is not None:
        for name, imp in zip(feature_names, importances):
            feature_importance_list.append({
                "feature": name,
                "importance": float(imp)
            })
        # sort by importance desc and keep top 10
        feature_importance_list = sorted(
            feature_importance_list, key=lambda x: x["importance"], reverse=True
        )[:10]

    return jsonify({
        "status": "success",
        "accuracy": round(float(acc) * 100, 2),
        "total_samples": total_samples,
        "malignant_count": malignant_count,
        "benign_count": benign_count,
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "feature_importance": feature_importance_list
    })


if __name__ == "__main__":
    app.run(debug=True)
