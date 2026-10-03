import litserve as ls
import torch
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"

models = {"text-classification": {}}


class Prompt(BaseModel):
    prompt: str


class Result(BaseModel):
    prompt: str
    sentiment: str


class TextClassifierAPI(ls.LitAPI):
    def setup(self, device):
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
            .eval()
            .to(device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def decode_request(self, request: Prompt, context=None, **kwargs):
        if context is None:
            raise ValueError("context cannot be None")
        tokens = self.tokenizer(
            request.prompt, padding=True, truncation=True, return_tensors="pt"
        )
        tokens = tokens.to(self.device)

        context["prompt"] = request.prompt

        return tokens

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.model(**x)
            logits = outputs.logits

        predictions = logits.softmax(dim=-1)
        predicted_class_ids = predictions.argmax(dim=-1)
        sentiment = self.model.config.id2label[predicted_class_ids[0].item()]

        return sentiment

    def encode_response(self, output, context=None, **kwargs):
        if context is None:
            raise ValueError("context cannot be None")
        return Result(sentiment=output, prompt=context["prompt"])


if __name__ == "__main__":
    api = TextClassifierAPI()
    server = ls.LitServer(api, accelerator="cuda")
    server.run(port=8000, generate_client_file=False)
