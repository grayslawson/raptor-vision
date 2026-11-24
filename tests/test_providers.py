"""Provider config, call budget, per-model cooldown (ported from Osprey test_assist_models / test_assist_provider_cooldown)."""

from __future__ import annotations

from typing import Any

import pytest

from raptor_vision.providers import Budget, ChatProvider, ProviderError, build_provider, parse_provider
from raptor_vision.tasks import TaskConfig, parse_task

from .conftest import Stub, chat


def _provider(url: str, now: list[float], **raw: Any) -> Any:
    cfg = parse_provider("gw", {"kind": "decisions", "base_url": url, "path": "/decisions", **raw})
    return build_provider("gw", cfg, clock=lambda: now[0])


def _ask(p: Any, model: str) -> None:
    p.decide(model, ["state"], {"q": {"type": "noul", "instructions": "?"}})


def test_daily_cap_skips_the_model_until_the_next_utc_day(stub: Any) -> None:
    replies = [(429, {"error": {"message": "litellm.RateLimitError: No deployments available for selected model"}})]
    s: Stub = stub(lambda req: replies.pop(0) if replies else (200, {"answers": {}}))
    now = [1_800_000_000.0 + 3600]  # an hour into a UTC day
    p = _provider(s.url, now)
    with pytest.raises(ProviderError):
        _ask(p, "capped")
    with pytest.raises(ProviderError, match="cooling down"):
        _ask(p, "capped")
    _ask(p, "other")  # another model on the same provider is unaffected
    assert [c["model"] for c in s.calls] == ["capped", "other"]
    now[0] += 86400 - 3600
    _ask(p, "capped")
    assert [c["model"] for c in s.calls] == ["capped", "other", "capped"]


def test_rate_limit_backs_off_and_doubles_then_resets_on_success(stub: Any) -> None:
    replies = [(429, {"error": {"message": "rate limited"}}), (502, {"error": {"message": "bad gateway"}})]
    s: Stub = stub(lambda req: replies.pop(0) if replies else (200, {"answers": {}}))
    now = [1_800_000_000.0]
    p = _provider(s.url, now)
    with pytest.raises(ProviderError):
        _ask(p, "m")
    with pytest.raises(ProviderError, match="cooling down"):
        _ask(p, "m")
    now[0] += 61
    with pytest.raises(ProviderError, match="http 502"):
        _ask(p, "m")
    now[0] += 61  # backoff doubled to 120 s
    with pytest.raises(ProviderError, match="cooling down"):
        _ask(p, "m")
    now[0] += 60
    _ask(p, "m")
    _ask(p, "m")
    assert len(s.calls) == 4


def test_extra_body_and_chat_request_shape(stub: Any) -> None:
    s: Stub = stub(lambda req: (200, chat({"ok": True})))
    cfg = parse_provider("c", {"kind": "chat", "base_url": s.url, "private": True,
                               "extra_body": {"provider": {"data_collection": "deny"}}, "json_schema": True})
    assert cfg.private
    p = build_provider("c", cfg)
    assert isinstance(p, ChatProvider)
    reply = p.complete("m1", "sys", "hello", [b"jpeg"], {"type": "object"})
    assert reply.data == {"ok": True} and reply.tokens_in == 10 and reply.model == "stub"
    req = s.calls[0]
    assert req["provider"] == {"data_collection": "deny"} and req["temperature"] == 0
    assert req["response_format"]["json_schema"]["name"] == "raptor_vision"
    assert req["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_key_from_env_and_unreadable_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RV_KEY", "secret")
    p = build_provider("p", parse_provider("p", {"kind": "chat", "base_url": "http://a", "api_key_env": "RV_KEY"}))
    assert p._key() == "secret"  # type: ignore[union-attr]
    bad = build_provider("p", parse_provider("p", {"kind": "chat", "base_url": "http://a", "api_key_file": "/nope/key"}))
    with pytest.raises(ProviderError, match="key file unreadable"):
        bad._key()  # type: ignore[union-attr]


def test_budget_window() -> None:
    now = [0.0]
    b = Budget(2, 3, lambda: now[0])
    b.record()
    b.record()
    assert not b.allow() and b.left == 0
    now[0] = 3601
    assert b.allow() and b.calls_today == 2 and b.left == 1
    b.record()
    now[0] = 3700
    assert not b.allow()  # day cap
    now[0] = 100000
    assert b.allow() and b.calls_today == 0


@pytest.mark.parametrize("raw", [
    {"kind": "chat"}, {"kind": "x", "base_url": "http://a"}, {"kind": "chat", "base_url": "ftp://a"},
    {"kind": "chat", "base_url": "http://a", "bogus": 1}, {"kind": "chat", "base_url": "http://a", "timeout_sec": 0},
    {"kind": "chat", "base_url": "http://a", "timeout_sec": 301}, {"kind": "chat", "base_url": "http://a", "allow_frames": "yes"},
    {"kind": "decisions", "base_url": "http://a", "json_schema": True},
    {"kind": "chat", "base_url": "http://a", "max_calls_per_hour": 0},
    {"kind": "chat", "base_url": "http://a", "headers": {"Authorization": "x"}},
    {"kind": "chat", "base_url": "http://a", "path": "chat"},
    {"kind": "chat", "base_url": "http://a", "private": "yes"}, {"kind": "chat", "base_url": "http://a", "extra_body": []},
])
def test_parse_provider_rejects(raw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_provider("p", raw)


def test_parse_provider_defaults() -> None:
    c = parse_provider("p", {"kind": "decisions", "base_url": "http://a/"})
    assert (c.path, c.base_url, c.allow_frames, c.private, c.extra_body, c.timeout_sec, c.max_calls_per_hour, c.max_calls_per_day) == (
        "/decisions", "http://a", False, False, {}, 30, 120, 2000)
    assert parse_provider("p", {"kind": "chat", "base_url": "http://a"}).path == "/chat/completions"


@pytest.mark.parametrize("raw", [
    {"mode": "model"}, {"mode": "both", "provider": "p"}, {"mode": "nope"}, {"min_confidence": 1.5}, {"send_frames": 1},
    {"zzz": 1}, {"threshold": 2}, {"fallback": [{"model": "m", "x": 1}]}, {"fallback": [{"model": "m", "threshold": -1}]},
])
def test_parse_task_rejects(raw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_task("survey.label", raw)


def test_parse_task() -> None:
    assert parse_task("t", {}) == TaskConfig()
    t = parse_task("t", {"mode": "both", "provider": "p", "model": "m", "send_frames": True, "min_confidence": 0.5,
                         "threshold": 0.4, "fallback": [{"model": "m2", "threshold": 0.3}, {"model": "m3"}]})
    assert (t.mode, t.send_frames, t.min_confidence, t.threshold) == ("both", True, 0.5, 0.4)
    assert t.fallback == (("m2", 0.3), ("m3", None))
