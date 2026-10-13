import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Literal, cast, get_args

import litserve as ls
import torch
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
Accelerator = Literal["cpu", "cuda", "mps", "auto"]
ACCELERATORS: tuple[str, ...] = get_args(Accelerator)
LOGGER_NAME = os.environ.get("LOGGER_NAME", "classifier")
LOG_FORMAT = "[%(levelname)s] %(asctime)s %(processName)s %(name)s %(message)s"


def read_log_level() -> str:
    level = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    if level not in logging.getLevelNamesMapping():
        raise ValueError(f"LOG_LEVEL must be a valid logging level, got {level!r}")
    return level


def configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(level)

    logger.propagate = False
    return logger


LOG_LEVEL = read_log_level()
logger = configure_logging(LOG_LEVEL)


class Prompt(BaseModel):
    prompt: str


class Result(BaseModel):
    prompt: str
    sentiment: str


class TextClassifierAPI(ls.LitAPI):
    def __init__(self, torch_num_threads: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.torch_num_threads = torch_num_threads

    def setup(self, device):
        if self.torch_num_threads > 0:
            torch.set_num_threads(self.torch_num_threads)

        start_time = time.perf_counter()
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
            .eval()
            .to(device)
        )
        end_time = time.perf_counter()
        model_load_elapsed_time = end_time - start_time

        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.total_requests = 0
        self.total_batches = 0
        self.time_of_last_report = time.monotonic()

        logger.info(
            "device: %s, pytorch threads: %d, model load time: %.6f seconds",
            device,
            torch.get_num_threads(),
            model_load_elapsed_time,
        )

    def decode_request(self, request: Prompt, context=None, **kwargs):
        if context is None:
            raise ValueError("context cannot be None")

        context["prompt"] = request.prompt

        return request.prompt

    def batch(self, inputs):
        self.total_requests += len(inputs)
        self.total_batches += 1

        now = time.monotonic()
        elapsed = now - self.time_of_last_report

        if elapsed >= 2.0:
            logger.info(
                "batch stats: requests=%d batches=%d avg_batch_size=%.2f "
                "max_batch_size=%d window=%.2fs",
                self.total_requests,
                self.total_batches,
                self.total_requests / self.total_batches,
                self.max_batch_size,
                elapsed,
            )

            self.total_requests = 0
            self.total_batches = 0
            self.time_of_last_report = now

        return inputs

    def predict(self, x, **kwargs):
        logger.debug("predicting")
        inputs = [x] if self.check_single_mode() else x

        try:
            results = self.classify_batch(inputs)
        except Exception:
            logger.exception(
                "classify_batch failed: batch_size=%d device=%s",
                len(inputs),
                self.device,
            )
            raise

        return results[0] if self.check_single_mode() else results

    def encode_response(self, output, context=None, **kwargs):
        if context is None:
            raise ValueError("context cannot be None")
        return Result(sentiment=output, prompt=context["prompt"])

    def check_single_mode(self):
        return self.max_batch_size == 1

    def classify_batch(self, inputs):
        tokens = self.tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        )
        tokens = tokens.to(self.device)

        with torch.no_grad():
            outputs = self.model(**tokens)
            logits = outputs.logits

        predicted_class_ids = logits.argmax(dim=-1)
        predicted_class_ids_list = predicted_class_ids.tolist()

        sentiments = [
            self.model.config.id2label[predicted_class_id]
            for predicted_class_id in predicted_class_ids_list
        ]

        return sentiments


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    max_batch_size: int
    batch_timeout: float
    accelerator: Accelerator
    num_api_servers: int
    workers_per_device: int
    torch_num_threads: int


def read_int(name: str, default: str, min_value: int) -> int:
    variable = os.environ.get(name, default)
    try:
        int_value = int(variable)
    except ValueError as e:
        raise ConfigError(f"{name} must be an integer, got {variable!r}") from e

    if int_value < min_value:
        raise ConfigError(f"{name} must be >= {min_value}, got {int_value}")

    return int_value


def read_float(name: str, default: str, min_value: float) -> float:
    variable = os.environ.get(name, default)
    try:
        float_value = float(variable)
    except ValueError as e:
        raise ConfigError(f"{name} must be a float, got {variable!r}") from e

    if not math.isfinite(float_value) or float_value < min_value:
        raise ConfigError(
            f"{name} must be finite and >= {min_value}, got {float_value}"
        )

    return float_value


def read_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    variable = os.environ.get(name, default)
    if variable in allowed:
        return variable

    raise ConfigError(f"{name} must be one of {', '.join(allowed)}, got {variable!r}")


def load_config() -> Config:
    max_batch_size = read_int("MAX_BATCH_SIZE", "1", 1)
    num_api_servers = read_int("NUM_API_SERVERS", "1", 1)
    workers_per_device = read_int("WORKERS_PER_DEVICE", "1", 1)
    torch_num_threads = read_int("TORCH_NUM_THREADS", "0", 0)
    batch_timeout = read_float("BATCH_TIMEOUT", "0.0", 0.0)
    accelerator = read_choice("ACCELERATOR", "cpu", ACCELERATORS)

    return Config(
        max_batch_size=max_batch_size,
        num_api_servers=num_api_servers,
        workers_per_device=workers_per_device,
        batch_timeout=batch_timeout,
        accelerator=cast(Accelerator, accelerator),
        torch_num_threads=torch_num_threads,
    )


if __name__ == "__main__":
    try:
        config = load_config()
    except ConfigError as e:
        sys.exit(str(e))

    print(config)
    logger.info("Config: %s", config)

    api = TextClassifierAPI(
        max_batch_size=config.max_batch_size,
        batch_timeout=config.batch_timeout,
        torch_num_threads=config.torch_num_threads,
    )
    server = ls.LitServer(
        api,
        accelerator=config.accelerator,
        workers_per_device=config.workers_per_device,
    )
    server.run(
        port=8000,
        generate_client_file=False,
        num_api_servers=config.num_api_servers,
        log_level=LOG_LEVEL.lower(),
    )
