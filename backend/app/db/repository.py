from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Protocol, runtime_checkable
from urllib.parse import unquote, urlparse, urlunparse

from .schema import POSTGRES_SCHEMA_SQL

try:
    import psycopg
    from psycopg import sql
except ImportError as exc:  # pragma: no cover - environment issue
    raise RuntimeError(
        "PostgreSQL backend requires psycopg. Install `psycopg[binary]` before starting the simulator."
    ) from exc


@runtime_checkable
class TrafficRepository(Protocol):
    def save_light_config(self, intersection_id: int, incoming_from: int, green: float, yellow: float, red: float) -> None: ...

    def load_light_config(self) -> Dict[tuple[int, int], tuple[float, float, float]]: ...

    def save_metrics_batch(self, rows: Iterable[tuple[float, int, int, float, float, int, float, float, int, float, float]]) -> None: ...


@dataclass
class PostgresRepository:
    database_url: str

    def __post_init__(self) -> None:
        self._ensure_database_exists()
        self._init_db()

    def _connect(self):
        return psycopg.connect(self.database_url)

    def _parsed_url(self):
        return urlparse(self.database_url)

    def _database_name(self) -> str:
        parsed = self._parsed_url()
        return unquote(parsed.path.lstrip("/"))

    def _admin_database_url(self) -> str:
        parsed = self._parsed_url()
        return urlunparse(parsed._replace(path="/postgres"))

    def _ensure_database_exists(self) -> None:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                conn.commit()
            return
        except Exception as exc:
            message = str(exc).lower()
            if "does not exist" not in message and "unknown database" not in message:
                raise

        db_name = self._database_name()
        if not db_name:
            raise RuntimeError("DATABASE_URL must include a database name")

        admin_url = self._admin_database_url()
        with psycopg.connect(admin_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))

    def _init_db(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(POSTGRES_SCHEMA_SQL)
            conn.commit()

    def save_light_config(self, intersection_id: int, incoming_from: int, green: float, yellow: float, red: float) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO traffic_light_config(intersection_id, incoming_from, green, yellow, red, updated_at)
                    VALUES(%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(intersection_id, incoming_from) DO UPDATE SET
                        green=EXCLUDED.green,
                        yellow=EXCLUDED.yellow,
                        red=EXCLUDED.red,
                        updated_at=EXCLUDED.updated_at
                    """,
                    (intersection_id, incoming_from, green, yellow, red, time.time()),
                )
            conn.commit()

    def load_light_config(self) -> Dict[tuple[int, int], tuple[float, float, float]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT intersection_id, incoming_from, green, yellow, red FROM traffic_light_config"
                )
                rows = cur.fetchall()
        return {
            (int(r[0]), int(r[1])): (float(r[2]), float(r[3]), float(r[4]))
            for r in rows
        }

    def save_metrics_batch(self, rows: Iterable[tuple[float, int, int, float, float, int, float, float, int, float, float]]) -> None:
        buffered = list(rows)
        if not buffered:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO metrics(timestamp, intersection_id, incoming_from, mc_avg_speed, mc_avg_density, mc_queue_length, car_avg_speed, car_avg_density, car_queue_length, local_imbalance, global_imbalance)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    buffered,
                )
            conn.commit()


def create_repository(database_url: str) -> TrafficRepository:
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Fill DB_PASSWORD or set DATABASE_URL in .env before starting the backend."
        )
    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise RuntimeError("Only PostgreSQL database URLs are supported.")
    return PostgresRepository(database_url)
