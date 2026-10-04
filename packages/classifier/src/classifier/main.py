import time

import litserve as ls
import torch
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"


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
        self.total_requests = 0
        self.total_batches = 0
        self.time_of_last_report = time.monotonic()

    def decode_request(self, request: Prompt, context=None, **kwargs):
        if context is None:
            raise ValueError("context cannot be None")

        context["prompt"] = request.prompt

        return request.prompt

    def batch(self, inputs):
        self.total_requests += len(inputs)
        self.total_batches += 1

        if time.monotonic() - self.time_of_last_report >= 2.0:
            avg_batch_size = self.total_requests // self.total_batches
            print(f"Average batch size: {avg_batch_size}")

            self.total_requests = 0
            self.total_batches = 0

        tokens = self.tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        )
        tokens = tokens.to(self.device)

        return tokens

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.model(**x)
            logits = outputs.logits

        predictions = logits.softmax(dim=-1)
        predicted_class_ids = predictions.argmax(dim=-1)
        predicted_class_ids_list = predicted_class_ids.tolist()

        sentiments = [
            self.model.config.id2label[predicted_class_id]
            for predicted_class_id in predicted_class_ids_list
        ]

        return sentiments

    def encode_response(self, output, context=None, **kwargs):
        if context is None:
            raise ValueError("context cannot be None")
        return Result(sentiment=output, prompt=context["prompt"])


if __name__ == "__main__":
    api = TextClassifierAPI(max_batch_size=8, batch_timeout=0.001)
    server = ls.LitServer(api, accelerator="cuda")
    server.run(port=8000, generate_client_file=False, num_api_servers=2)
