from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ApiError(RuntimeError):
    """Raised when backend API returns an error."""


@dataclass
class TrafficApiClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 5.0

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = Request(self._url(path), data=body, headers=headers, method=method.upper())
        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raise ApiError(f"{method} {path} failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise ApiError(f"Cannot connect to backend at {self.base_url}: {exc.reason}") from exc

        if not raw:
            return {}
        return json.loads(raw)

    def get_network(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/v1/network")

    def get_state(self, intersection_id: int) -> dict[str, Any]:
        return self.request_json("GET", f"/api/v1/state/{intersection_id}")

    def get_reward_metrics(self, intersection_id: int) -> dict[str, Any]:
        return self.request_json("GET", f"/api/v1/reward_metrics/{intersection_id}")

    def post_action(self, intersection_id: int, action: int) -> dict[str, Any]:
        return self.request_json("POST", f"/api/v1/action/{intersection_id}", {"action": int(action)})

    def reset(self) -> dict[str, Any]:
        return self.request_json("POST", "/api/v1/reset")

