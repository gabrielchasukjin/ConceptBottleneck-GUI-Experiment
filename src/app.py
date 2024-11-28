import psutil
import platform
import torch
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
        subprocess.run("python", "train_CBL.py --automatic_concept_correction")
        # Will eventually configure this last process to pick the best epoch model
        # This will require us to save the accuracies when the final layer is being trained
        subprocess.run("python", "train_FL.py --cbl_path mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc_epoch_8.pt")

        return jsonify({"message": f"Training completed on {hardware}!"})

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Training failed: {e.stderr}"}), 500


# Test endpoint
@app.route("/test", methods=["POST"])
def test_model():
    data = request.json
    input_text = data.get("input", "")
    if not input_text:
        return jsonify({"error": "Input text is required"}), 400
    try:

        # simulate classification and concept activation
        classification = "positive" if "good" in input_text else "negative"

        activations = {
            "Concept A": 0.8,
            "Concept B": 0.5,
            "Concept C": 0.3,
            "Concept D": 0.9,
        }
        return jsonify({"classification": classification, "activations": activations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)

