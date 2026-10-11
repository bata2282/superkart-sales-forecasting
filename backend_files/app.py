import os
import gzip
import joblib
import shutil
import tempfile
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "superkart_model.joblib"
)

model = joblib.load(MODEL_PATH)

# Expected input features
FEATURES = [
    "Product_Id",
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "message": "SuperKart prediction API is running"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Request must contain a JSON object"
            }), 400

        missing_features = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        # Create input DataFrame in the expected column order
        input_df = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        # Predict sales
        prediction = model.predict(input_df)

        return jsonify({
            "predicted_sales": round(float(prediction[0]), 2)
        }), 200

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": "Invalid input data",
            "details": str(error)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Unable to generate prediction"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
