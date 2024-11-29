import psutil
import platform
import torch
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess


app = Flask(__name__)
CORS(app)  # Enables cross-origin requests


# Train endpoint
@app.route("/train", methods=["POST"])
def train_model():
    data = request.json
    hardware = data.get("hardware", "Local Hardware")  # default to local hardware
    try:
        if hardware == "Local Hardware":
            print("Training on local hardware...")
        elif hardware == "DSMLP":
            print("Training on DSMLP...")
        else:
            return jsonify({"error": "Invalid hardware option"}), 400

        subprocess.run(["python","get_concept_labels.py"])
        subprocess.run(["python", "train_CBL.py", "--automatic_concept_correction"])
        # Will eventually configure this last process to pick the best epoch model
        # This will require us to save the accuracies when the final layer is being trained
        subprocess.run(["python", "train_FL.py", "--cbl_path", "mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc_epoch_8.pt"])

        return jsonify({"message": f"Training completed on {hardware}!"})

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Training failed: {e.stderr}"}), 500


@app.route("/test", methods=["POST"])
def test_model():
    data = request.json
    input_text = data.get("input", "")
    if not input_text:
        return jsonify({"error": "Input text is required"}), 400

    try:
        # Call the test_model.py script with the input text
        result = subprocess.run(
            ["python", "test_model.py", input_text],
            capture_output=True,
            text=True
        )

        # Check for errors during script execution
        if result.returncode != 0:
            print(f"Error running test_model.py: {result.stderr}")  # Log the error
            return jsonify({"error": f"Error in test_model.py: {result.stderr}"}), 500

        # Parse the JSON response from test_model.py
        response_data = json.loads(result.stdout)
        return jsonify(response_data)

    except Exception as e:
        print(f"Exception in /test endpoint: {str(e)}")  # Log any exceptions
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)

