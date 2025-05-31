from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import uvicorn
import json
import numpy as np

# === Load model & tokenizer ===
MODEL_PATH = "./best_model"
label_counts = 22
k = 100
weights = 1.0 / np.log(label_counts + k)
weights = weights / np.max(weights)  # normalize to [0, 1]
loss_weights = torch.tensor(weights, dtype=torch.float32)

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH,loss_weights)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model.eval()

# === FastAPI app ===
app = FastAPI()

# === Define label mapping ===
with open("labels.json", "r") as f:
    label_names = json.load(f)

# === Input schema ===
class TextInput(BaseModel):
    text: str

# === Predict endpoint ===
@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(
        input.text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()

    threshold = 0.5
    predicted = [label_names[i] for i, p in enumerate(probs) if p >= threshold]
    return {"predicted_labels": predicted, "scores": probs}

# === To run ===
# uvicorn app:app --reload
