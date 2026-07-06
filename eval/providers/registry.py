"""
Lazily-constructed, cached provider instances, keyed by provider name. SDKs
are imported lazily inside each branch so that using only one provider
doesn't require every provider's SDK to be installed.

Tests should use set_provider_override() rather than monkeypatching SDK
clients directly -- see tests/test_smoke.py.
"""

from .base import Provider

_instances: dict[str, Provider] = {}
_overrides: dict[str, Provider] = {}


def get_provider(name: str) -> Provider:
    if name in _overrides:
        return _overrides[name]

    if name not in _instances:
        if name == "groq":
            from .groq_provider import GroqProvider
            _instances[name] = GroqProvider()
        elif name == "openai":
            from .openai_provider import OpenAICompatProvider
            _instances[name] = OpenAICompatProvider()
        elif name == "sarvam":
            from .sarvam_provider import SarvamProvider
            _instances[name] = SarvamProvider()
        else:
            raise ValueError(f"Unknown provider: {name!r}")

    return _instances[name]


def set_provider_override(name: str, provider: Provider) -> None:
    """Test hook: force get_provider(name) to return this instance instead
    of constructing a real SDK client."""
    _overrides[name] = provider


def clear_provider_overrides() -> None:
    _overrides.clear()
