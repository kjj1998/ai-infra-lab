from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"

models = {"text-classification": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    models["text-classification"]["model"] = (
        AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    )
    models["text-classification"]["tokenizer"] = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models["text-classification"]["model"].to(device)
    models["text-classification"]["model"].eval()
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


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
    model = models["text-classification"]["model"]
    tokenizer = models["text-classification"]["tokenizer"]
    tokens = tokenizer(
        prompt.prompt, padding=True, truncation=True, return_tensors="pt"
    )
    tokens = tokens.to(model.device)

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits

    predictions = logits.softmax(dim=-1)
    predicted_class_ids = predictions.argmax(dim=-1)
    sentiment = model.config.id2label[predicted_class_ids[0].item()]

    return Result(prompt=prompt.prompt, sentiment=sentiment)
