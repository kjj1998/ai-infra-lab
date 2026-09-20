import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Prompt(BaseModel):
    prompt: str


class Result(BaseModel):
    prompt: str
    sentiment: str


@app.put("/predict")
def predict(prompt: Prompt) -> Result:
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)

    tokens = tokenizer(
        prompt.prompt, padding=True, truncation=True, return_tensors="pt"
    )
    outputs = model(**tokens)

    with torch.no_grad():
        logits = outputs.logits

    predictions = logits.softmax(dim=-1)
    predicted_class_ids = predictions.argmax(dim=-1)
    sentiment = model.config.id2label[predicted_class_ids[0].item()]

    return Result(prompt=prompt.prompt, sentiment=sentiment)
