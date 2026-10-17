import pytest
from classifier.main import (
    Config,
    ConfigError,
    load_config,
    read_choice,
    read_float,
    read_int,
)

INT_ENV_VAR = "TEST_READ_INT"
FLOAT_ENV_VAR = "TEST_READ_FLOAT"
CHOICE_ENV_VAR = "TEST_READ_CHOICE"


def test_read_int_returns_env_value(monkeypatch):
    monkeypatch.setenv(INT_ENV_VAR, "8")

    assert read_int(INT_ENV_VAR, default="1", min_value=1) == 8


def test_read_int_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv(INT_ENV_VAR, raising=False)

    assert read_int(INT_ENV_VAR, default="4", min_value=1) == 4


def test_read_int_accepts_min_value(monkeypatch):
    monkeypatch.setenv(INT_ENV_VAR, "0")

    assert read_int(INT_ENV_VAR, default="1", min_value=0) == 0


@pytest.mark.parametrize(
    ("raw", "expected_message"),
    [
        ("abc", "must be an integer"),
        ("1.5", "must be an integer"),
        ("", "must be an integer"),
        ("-1", "must be >= 0"),
    ],
)
def test_read_int_rejects_invalid_values(monkeypatch, raw, expected_message):
    monkeypatch.setenv(INT_ENV_VAR, raw)

    with pytest.raises(ConfigError, match=expected_message):
        read_int(INT_ENV_VAR, default="1", min_value=0)


def test_read_int_chains_original_error(monkeypatch):
    monkeypatch.setenv(INT_ENV_VAR, "abc")

    with pytest.raises(ConfigError) as exc_info:
        read_int(INT_ENV_VAR, default="1", min_value=0)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_read_float_returns_env_value(monkeypatch):
    monkeypatch.setenv(FLOAT_ENV_VAR, "8.0")

    assert read_float(FLOAT_ENV_VAR, default="1.0", min_value=1.0) == 8.0


def test_read_float_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv(FLOAT_ENV_VAR, raising=False)

    assert read_float(FLOAT_ENV_VAR, default="0.0", min_value=0.0) == 0.0


def test_read_float_accepts_min_value(monkeypatch):
    monkeypatch.setenv(FLOAT_ENV_VAR, "0.0")

    assert read_float(FLOAT_ENV_VAR, default="1.0", min_value=0.0) == 0


@pytest.mark.parametrize(
    ("raw", "expected_message"),
    [
        ("abc", "must be a float"),
        ("", "must be a float"),
        ("-1", "must be finite and >= 0.0"),
        ("nan", "must be finite and >= 0.0"),
        ("inf", "must be finite and >= 0.0"),
        ("-inf", "must be finite and >= 0.0"),
    ],
)
def test_read_float_rejects_invalid_values(monkeypatch, raw, expected_message):
    monkeypatch.setenv(FLOAT_ENV_VAR, raw)

    with pytest.raises(ConfigError, match=expected_message):
        read_float(FLOAT_ENV_VAR, default="1.0", min_value=0.0)


def test_read_float_chains_original_error(monkeypatch):
    monkeypatch.setenv(FLOAT_ENV_VAR, "abc")

    with pytest.raises(ConfigError) as exc_info:
        read_int(FLOAT_ENV_VAR, default="1", min_value=0)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_read_choice_returns_env_value(monkeypatch):
    monkeypatch.setenv(CHOICE_ENV_VAR, "a")

    assert read_choice(CHOICE_ENV_VAR, default="b", allowed=("a", "b")) == "a"


def test_read_choice_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv(CHOICE_ENV_VAR, raising=False)

    assert read_choice(CHOICE_ENV_VAR, default="b", allowed=("a", "b")) == "b"


@pytest.mark.parametrize(
    ("raw", "expected_message"),
    [
        ("c", "must be one of a, b"),
        ("A", "must be one of a, b"),
    ],
)
def test_read_choice_rejects_disallowed_value(monkeypatch, raw, expected_message):
    monkeypatch.setenv(CHOICE_ENV_VAR, raw)

    with pytest.raises(ConfigError, match=expected_message):
        read_choice(CHOICE_ENV_VAR, default="b", allowed=("a", "b"))


def test_load_config_defaults(clean_config_env):
    config = load_config()

    assert config == Config(
        max_batch_size=1,
        batch_timeout=0.0,
        accelerator="cpu",
        num_api_servers=1,
        workers_per_device=1,
        torch_num_threads=0,
    )


def test_load_config_rejects_bad_value(clean_config_env):
    clean_config_env.setenv("MAX_BATCH_SIZE", "0")

    with pytest.raises(ConfigError, match="MAX_BATCH_SIZE"):
        load_config()
