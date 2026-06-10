from __future__ import annotations

from dataclasses import dataclass, field

from db.repository import PostgresRepository, create_repository


@dataclass
class FakeState:
    light_configs: dict[tuple[int, int], tuple[float, float, float]] = field(default_factory=dict)
    metrics: list[tuple] = field(default_factory=list)


class FakeCursor:
    def __init__(self, state: FakeState):
        self.state = state
        self.rows: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        normalized = " ".join(str(query).split())
        if normalized == "SELECT 1":
            return None
        if normalized.startswith("CREATE DATABASE "):
            return None
        if normalized.startswith("INSERT INTO traffic_light_config"):
            intersection_id, incoming_from, green, yellow, red, *_ = params
            self.state.light_configs[(int(intersection_id), int(incoming_from))] = (
                float(green),
                float(yellow),
                float(red),
            )
            return None
        if normalized.startswith("SELECT intersection_id, incoming_from, green, yellow, red FROM traffic_light_config"):
            self.rows = [
                (iid, incoming, values[0], values[1], values[2])
                for (iid, incoming), values in self.state.light_configs.items()
            ]
            return None
        if normalized.startswith("INSERT INTO metrics"):
            self.state.metrics.append(tuple(params))
            return None
        return None

    def executemany(self, query, rows):
        for params in rows:
            self.execute(query, params)

    def fetchall(self):
        return list(self.rows)


class FakeConnection:
    def __init__(self, state: FakeState):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return FakeCursor(self.state)

    def commit(self):
        return None


def test_postgres_repository_roundtrip(monkeypatch):
    state = FakeState()

    def fake_connect(url, autocommit=False):
        return FakeConnection(state)

    monkeypatch.setattr("db.repository.psycopg.connect", fake_connect)

    repo = PostgresRepository("postgresql://traffic_user:pw@localhost:5432/traffic_simulator?sslmode=prefer")
    repo.save_light_config(1, 2, 11.0, 3.0, 16.0)
    repo.save_metrics_batch([(1.0, 1, 2, 4.0, 5.0, 6, 7.0, 8.0, 9, 10.0, 11.0)])

    loaded = repo.load_light_config()
    assert loaded[(1, 2)] == (11.0, 3.0, 16.0)
    assert state.metrics[0] == (1.0, 1, 2, 4.0, 5.0, 6, 7.0, 8.0, 9, 10.0, 11.0)


def test_create_repository_rejects_non_postgres_urls():
    try:
        create_repository("mysql://example/db")
    except RuntimeError as exc:
        assert "Only PostgreSQL database URLs" in str(exc)
    else:
        raise AssertionError("Expected create_repository to reject non-PostgreSQL URLs")
