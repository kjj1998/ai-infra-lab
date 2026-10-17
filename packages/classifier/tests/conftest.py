import pytest

CONFIG_ENV_VARS = (
    "MAX_BATCH_SIZE",
    "NUM_API_SERVERS",
    "WORKERS_PER_DEVICE",
    "TORCH_NUM_THREADS",
    "BATCH_TIMEOUT",
    "ACCELERATOR",
)


@pytest.fixture
def clean_config_env(monkeypatch):
    """Remove all config env vars so load_config() sees only its defaults."""
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch
