import json
import time

from riskpulse.bus import InProcessSubscriber, SignalBus, SQLiteTailSubscriber
from riskpulse.store import Store


def test_store_filters_and_jsonl(tmp_path, make_signal):
    store = Store(tmp_path / "t.db", jsonl=tmp_path / "s.jsonl")
    store.add_signals([make_signal(0), make_signal(1, impact_score=4.0, event_type="Earnings")])
    store.add_signals([make_signal(0)])                                   # idempotent on signal_id
    assert store.signal_count() == 2
    assert [s["signal_id"] for s in store.signals(min_impact=7)] == ["s0"]
    assert [s["signal_id"] for s in store.signals(event_type="Earnings")] == ["s1"]
    lines = (tmp_path / "s.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["entity"]["id"] == "BA"


def test_bus_filters_and_isolates_failing_subscribers(make_signal):
    bus, got = SignalBus(), []
    InProcessSubscriber(bus).subscribe("ok", got.extend, lambda s: s.impact_score > 7)
    bus.subscribe("broken", lambda batch: 1 / 0)
    bus.publish([make_signal(0), make_signal(1, impact_score=3.0)])
    assert [s.signal_id for s in got] == ["s0"]
    assert {s["subscriber"]: s["failures"] for s in bus.status()} == {"ok": 0, "broken": 1}


def test_sqlite_tail_subscriber_sees_new_rows(tmp_path, make_signal):
    store = Store(tmp_path / "t.db", jsonl=False)
    got = []
    sub = SQLiteTailSubscriber(tmp_path / "t.db", from_start=True, poll_seconds=0.05)
    sub.subscribe("tail", got.extend)
    store.add_signals([make_signal(i) for i in range(3)])
    deadline = time.time() + 5
    while len(got) < 3 and time.time() < deadline:
        time.sleep(0.05)
    sub.close()
    assert sorted(s.signal_id for s in got) == ["s0", "s1", "s2"]
