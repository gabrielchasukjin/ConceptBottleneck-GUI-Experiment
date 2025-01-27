import warnings
import sys
import json
import numpy as np
import torch
import config as CFG
from datasets import load_dataset
from transformers import RobertaTokenizerFast
from modules import RobertaCBL

warnings.filterwarnings("ignore")

dataset = load_dataset("SetFit/sst2")
concept_set = CFG.concept_set["SetFit/sst2"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cbl_path = "mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc_epoch_8.pt"
concept_set_size = 208
dropout_rate = 0.1

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
full_model = RobertaCBL(concept_set_size, dropout_rate).to(device)
state_dict = torch.load(cbl_path, map_location=device)
full_model.load_state_dict(state_dict)
full_model.eval()

def preprocess_input(text, tokenizer, max_length=512):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {k: v.to(device) for k, v in tokens.items()}

def predict_full_model(text, tokenizer, model):
    tokens = preprocess_input(text, tokenizer)
    with torch.no_grad():
        outputs = model(tokens)
        
        # Get concepts and classification
        if isinstance(outputs, tuple):
            concepts, logits = outputs[0], outputs[1]
        else:
            concepts = outputs
            logits = model.final_layer(concepts)

        concepts = torch.relu(concepts)
        # Get classification prediction
        prediction = torch.sigmoid(logits).item()
        classification = "Positive" if prediction > 0.5 else "Negative"

    return concepts.cpu().numpy(), classification, prediction

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Input text is required"}))
        sys.exit(1)

    input_text = sys.argv[1]

    # Get predictions
    concepts, classification, confidence = predict_full_model(input_text, tokenizer, full_model)
    concept_activations = concepts[0]

    # Filter and sort concepts
    active_concepts = [
        (concept_set[i], float(concept_activations[i]))
        for i in range(len(concept_activations))
        if concept_activations[i] > 0
    ]
    active_concepts.sort(key=lambda x: x[1], reverse=True)

    top_n = 10
    top_concepts = active_concepts[:top_n]

    # Prepare JSON response
    response = {
        "input_text": input_text,
        "classification": classification,
        "confidence": round(confidence * 100, 2),
        "top_concepts": [{"concept": concept, "activation": round(value, 4)} for concept, value in top_concepts]
    }

    print(json.dumps(response))

    