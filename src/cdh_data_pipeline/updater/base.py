"""Pacing primitives shared by every archive client.

CHC bans clients that pull hard, and a throttle response carries a wait that is
not negotiable. Both behaviours live here rather than in a mission's client, so
raising worker count anywhere cannot raise the request rate.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Space out requests to a shared host, across threads.

    Exponential backoff only helps *after* a refusal; this keeps the steady-state
    request rate polite enough not to provoke one. The limiter is shared by every
    call through a given client, so raising worker count cannot raise the request
    rate above ``1 / min_interval`` -- concurrency and politeness are decoupled on
    purpose.
    """

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_at - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_at = now + self.min_interval

    def back_off(self, seconds: float) -> None:
        """Push the next allowed request out, after a throttle response."""
        with self._lock:
            self._next_at = max(self._next_at, time.monotonic() + seconds)


#: HTTP statuses that mean "slow down", as opposed to "this failed".
THROTTLE_STATUSES = {429, 503, 509}

#: How long to wait when the server throttles us and gives no Retry-After.
DEFAULT_BACKOFF = 60.0


def throttle_delay(exc, attempt: int, default: float = DEFAULT_BACKOFF) -> float | None:
    """Seconds to wait for a throttle response, or None if not a throttle.

    Honours ``Retry-After`` when the server sends it -- guessing shorter than the
    server asked is how a slow-down becomes a ban.
    """
    status = getattr(exc, "code", None)
    if status not in THROTTLE_STATUSES:
        return None
    retry_after = None
    headers = getattr(exc, "headers", None)
    if headers is not None:
        raw = headers.get("Retry-After")
        if raw:
            try:
                retry_after = float(raw)
            except (TypeError, ValueError):
                retry_after = None
    return retry_after if retry_after is not None else default * (attempt + 1)
