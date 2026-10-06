"""Sinks: SQLite (queried by the API and dashboard) and an append-only JSONL file.

runtime/signals.jsonl is the "write to a file" output channel: any downstream tool can tail it
without talking to the API. Modules also persist their own state here (weights, stress runs).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

from riskpulse.config import ensure_runtime
from riskpulse.schema import RiskSignal

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id TEXT UNIQUE, story_id TEXT, published_at TEXT,
    generated_at TEXT, entity_type TEXT, entity_id TEXT, event_type TEXT, sentiment REAL, impact REAL,
    source TEXT, source_type TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS ix_signals_entity ON signals(entity_id, id);
CREATE TABLE IF NOT EXISTS weights (ts TEXT, ticker TEXT, weight REAL, base_weight REAL, sentiment REAL,
    PRIMARY KEY (ts, ticker));
CREATE TABLE IF NOT EXISTS rebalance_log (ts TEXT, ticker TEXT, old_weight REAL, new_weight REAL, sentiment REAL,
    reason TEXT);
CREATE TABLE IF NOT EXISTS stress_runs (run_id TEXT PRIMARY KEY, ts TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


class Store:
    def __init__(self, path: Path | None = None, jsonl: Path | None | bool = True):
        rt = ensure_runtime()
        self.path = Path(path) if path else rt / "riskpulse.db"
        self.jsonl_path = (rt / "signals.jsonl") if jsonl is True else (Path(jsonl) if jsonl else None)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def _exec(self, sql: str, args=(), many: bool = False):
        with self._lock:
            cur = self.conn.executemany(sql, args) if many else self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    def _query(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, args).fetchall()

    # ------------------------------------------------------------------ signals
    def add_signals(self, signals: Iterable[RiskSignal]) -> int:
        rows = [(s.signal_id, s.story_id, s.published_at.isoformat(), s.generated_at.isoformat(), s.entity.type,
                 s.entity.id, s.event_type, s.sentiment_score, s.impact_score, s.source, s.source_type.value,
                 s.model_dump_json()) for s in signals]
        if not rows:
            return 0
        self._exec("INSERT OR IGNORE INTO signals (signal_id, story_id, published_at, generated_at, entity_type, "
                   "entity_id, event_type, sentiment, impact, source, source_type, payload) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows, many=True)
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.writelines(r[-1] + "\n" for r in rows)
        return len(rows)

    def signals(self, since_id: int = 0, limit: int = 200, entity_id: str | None = None,
                event_type: str | None = None, min_impact: float | None = None, since: str | None = None,
                ascending: bool = False) -> list[dict]:
        q, args = "SELECT id, payload FROM signals WHERE id > ?", [since_id]
        for col, val, op in (("entity_id", entity_id, "="), ("event_type", event_type, "="),
                             ("impact", min_impact, ">="), ("published_at", since, ">=")):
            if val is not None:
                q += f" AND {col} {op} ?"
                args.append(val)
        q += f" ORDER BY id {'ASC' if ascending else 'DESC'} LIMIT ?"
        args.append(limit)
        return [{"id": r["id"], **json.loads(r["payload"])} for r in self._query(q, args)]

    def max_signal_id(self) -> int:
        return self._query("SELECT COALESCE(MAX(id), 0) AS m FROM signals")[0]["m"]

    def signal_count(self) -> int:
        return self._query("SELECT COUNT(*) AS n FROM signals")[0]["n"]

    # ------------------------------------------------------------------ module A
    def add_weights(self, ts: str, rows: list[tuple[str, float, float, float]]) -> None:
        self._exec("INSERT OR REPLACE INTO weights VALUES (?,?,?,?,?)", [(ts, *r) for r in rows], many=True)

    def weights_history(self, max_snapshots: int = 2000) -> list[dict]:
        return [dict(r) for r in self._query(
            "SELECT * FROM weights WHERE ts IN (SELECT DISTINCT ts FROM weights ORDER BY ts DESC LIMIT ?) ORDER BY ts",
            (max_snapshots,))]

    def add_rebalance_log(self, rows: list[tuple]) -> None:
        if rows:
            self._exec("INSERT INTO rebalance_log VALUES (?,?,?,?,?,?)", rows, many=True)

    def rebalance_log(self, limit: int = 200) -> list[dict]:
        return [dict(r) for r in self._query("SELECT * FROM rebalance_log ORDER BY ts DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------------ module B
    def add_stress_run(self, run: dict) -> None:
        self._exec("INSERT OR REPLACE INTO stress_runs VALUES (?,?,?)", (run["run_id"], run["ts"], json.dumps(run)))

    def stress_runs(self, limit: int = 50) -> list[dict]:
        return [json.loads(r["payload"]) for r in
                self._query("SELECT payload FROM stress_runs ORDER BY ts DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------------ misc
    def put(self, k: str, v) -> None:
        self._exec("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v, default=str)))

    def get(self, k: str, default=None):
        rows = self._query("SELECT v FROM kv WHERE k = ?", (k,))
        return json.loads(rows[0]["v"]) if rows else default

    def reset(self) -> None:
        for t in ("signals", "weights", "rebalance_log", "stress_runs", "kv"):
            self._exec(f"DELETE FROM {t}")
        if self.jsonl_path and self.jsonl_path.exists():
            self.jsonl_path.unlink()
