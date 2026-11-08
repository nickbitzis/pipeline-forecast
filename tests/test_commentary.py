from types import SimpleNamespace

import pytest

from pipecast.commentary import TemplateWriter, build_facts, get_writer, money, pct, validate, write_commentary

FACTS = {"quarter": "FY26 Q4", "week": "7", "p10": "$6.53M", "p50": "$7,034,500", "share": "85%",
         "deals": "131", "top": [{"amount": "$392K", "p_close": "17%"}]}


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.messages = self
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)],
                               usage=SimpleNamespace(input_tokens=500, output_tokens=120))


def test_formatters():
    assert money(4_213_500) == "$4.21M" and money(850_400) == "$850K" and money(12) == "$12"
    assert pct(0.849) == "85%"


def test_validator_accepts_facts_and_their_roundings():
    text = "FY26 Q4, week 7: P50 $7.03M ($7.0M, $7,034,500), P10 $6.53M, 85% coverage, 131 deals, $392K at 17%."
    v = validate(text, FACTS)
    assert v["valid"], v["violations"]
    assert v["numbers_checked"] == 9


def test_validator_rejects_invented_numbers():
    v = validate("The quarter lands at $7.4M with 12 deals at risk.", FACTS)
    assert not v["valid"] and v["violations"] == ["$7.4M", "12"]


def test_validator_keeps_percent_and_plain_numbers_apart():
    assert not validate("We expect 131% growth.", FACTS)["valid"]
    assert not validate("Coverage was 85 points.", FACTS)["valid"]


def test_validator_ignores_labels_but_not_bare_numbers():
    assert validate("P10 and P90 for FY26 Q4.", FACTS)["valid"]
    assert not validate("Q4 is week 8.", FACTS)["valid"]


def test_template_commentary_validates_on_real_facts(engine):
    fc = engine.public_forecast()
    facts = build_facts(fc, engine.risk(limit=10), engine.backtest_report(), engine.reps)
    out = write_commentary(facts, TemplateWriter())
    assert out["valid"], out["violations"]
    assert out["numbers_checked"] > 10 and facts["p10"] in out["text"]


def test_claude_writer_with_fake_client_passes(engine):
    fc = engine.public_forecast()
    facts = build_facts(fc, engine.risk(limit=10), engine.backtest_report(), engine.reps)
    client = FakeClient(f"Commit {facts['p10']}. Most likely {facts['p50']}, range up to {facts['p90']}.")
    out = write_commentary(facts, get_writer("claude", client=client))
    assert out["provider"] == "claude" and out["valid"] and not out["fallback"]
    assert out["usage"]["output_tokens"] == 120
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5" and facts["p10"] in call["messages"][0]["content"]


def test_claude_text_with_made_up_number_falls_back(engine):
    fc = engine.public_forecast()
    facts = build_facts(fc, engine.risk(limit=10), engine.backtest_report(), engine.reps)
    out = write_commentary(facts, get_writer("claude", client=FakeClient("We will close $99.9M this quarter.")))
    assert out["fallback"] and out["provider"] == "template" and out["valid"]
    assert out["rejected_violations"] == ["$99.9M"]


def test_default_writer_is_offline(monkeypatch):
    monkeypatch.delenv("PIPECAST_LLM_PROVIDER", raising=False)
    assert get_writer().name == "template"


@pytest.mark.parametrize("text", ["", "No numbers at all."])
def test_text_without_numbers_is_valid(text):
    assert validate(text, FACTS)["valid"]
