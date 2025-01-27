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


# Add new hardware info endpoint
@app.route("/hardware-info", methods=["GET"])
def get_hardware_info():
    try:
        # Get CPU info
        cpu_info = platform.processor()
        if not cpu_info:  # Fallback if processor() returns empty string
            cpu_info = platform.machine()
        
        # Get RAM info (in GB, rounded to 2 decimal places)
        ram_gb = round(psutil.virtual_memory().total / (1024.0 ** 3), 2)
        ram_info = f"{ram_gb}GB"
        
        # Get GPU info
        gpu_info = "None"
        if platform.system() == "Darwin" and platform.processor() == "arm":
            # For Apple Silicon Macs
            gpu_info = "Apple Integrated GPU"
            if "M1" in cpu_info:
                gpu_info = "Apple M1 GPU"
            elif "M2" in cpu_info:
                gpu_info = "Apple M2 GPU"
            elif "M3" in cpu_info:
                gpu_info = "Apple M3 GPU"
        elif torch.cuda.is_available():
            # For NVIDIA GPUs
            gpu_info = torch.cuda.get_device_name(0)
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)
                gpu_info += f" ({round(gpu_mem, 2)}GB)"
            except:
                pass

        return jsonify({
            'cpu': cpu_info,
            'ram': ram_info,
            'gpu': gpu_info
        })
    except Exception as e:
        print(f"Error getting hardware info: {str(e)}")
        return jsonify({
            'cpu': 'Unknown',
            'ram': 'Unknown',
            'gpu': 'Unknown'
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)

