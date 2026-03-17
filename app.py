from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import os
import pickle
from utils import secure_filename

# If your prediction logic is inside prediction_pipeline.py
# you can import it here and use directly instead of loading pkl.
# Example:
# from prediction_pipeline import PredictionPipeline

app = Flask(__name__)
app.secret_key = "replace_with_random_secret"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"csv"}

# Feature order (must match training)
FEATURE_NAMES = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]

# Load pickled model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html", feature_names=FEATURE_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    """Handles single prediction and batch prediction via CSV"""

    # --- CSV Upload ---
    if "csvfile" in request.files and request.files["csvfile"].filename != "":
        file = request.files["csvfile"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                flash(f"CSV read error: {e}")
                return redirect(url_for("index"))

            missing = [c for c in FEATURE_NAMES if c not in df.columns]
            if missing:
                flash(f"Missing columns: {missing}")
                return redirect(url_for("index"))

            preds = model.predict(df[FEATURE_NAMES])
            df["prediction"] = preds
            return render_template(
                "index.html",
                feature_names=FEATURE_NAMES,
                table=df.to_html(classes="table table-striped", index=False)
            )

    # --- Single Form Prediction ---
    try:
        values = {name: request.form.get(name) for name in FEATURE_NAMES}
        if any(v is None or v.strip() == "" for v in values.values()):
            raise ValueError("All fields are required.")

        # Create dataframe row
        df = pd.DataFrame([values])

        # Convert numeric fields
        for col in ["carat", "depth", "table", "x", "y", "z"]:
            df[col] = df[col].astype(float)

        pred = model.predict(df)[0]

        return render_template(
            "index.html",
            feature_names=FEATURE_NAMES,
            single_prediction=pred
        )

    except Exception as e:
        flash(f"Error: {e}")
        return redirect(url_for("index"))


@app.route("/predict_json", methods=["POST"])
def predict_json():
    """API endpoint for JSON input"""
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON provided"}), 400

    if "instances" in data:
        X = pd.DataFrame(data["instances"], columns=FEATURE_NAMES)
    elif "features" in data:
        X = pd.DataFrame([data["features"]], columns=FEATURE_NAMES)
    else:
        return jsonify({"error": "JSON must contain 'features' or 'instances'"}), 400

    preds = model.predict(X)
    return jsonify({"predictions": preds.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
