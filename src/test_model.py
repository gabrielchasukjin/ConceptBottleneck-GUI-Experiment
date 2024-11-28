# test_model.py
import sys
import json

# Retrieve input text from the command-line argument
input_text = sys.argv[1]

classification = "positive" if "good" in input_text else "negative"
activations = {
    "Concept A": 0.8,
    "Concept B": 0.5,
    "Concept C": 0.3,
    "Concept D": 0.9,
}

output = {
    "classification": classification,
    "activations": activations
}
print(json.dumps(output))
