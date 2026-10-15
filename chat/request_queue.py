"""Process-wide fair FIFO admission queue for chat requests.

Why an app-level queue when vLLM already schedules?  The production recipe runs
``MAX_NUM_SEQS=1``, so vLLM serialises requests anyway -- but silently: a second
user just sees a frozen spinner and nobody can tell how long the line is.  This
queue gives every request a *ticket*, admits tickets strictly in arrival order
up to ``max_concurrency`` (default = the recipe's MAX_NUM_SEQS), lets each
waiting user see their position, rejects new work when the line is too long
instead of piling up, and caps each user at one in-flight request so a single
tab cannot monopolise the box.

Scope: single process.  Streamlit runs every browser session as a thread inside
one server process, so a module-level singleton (held via st.cache_resource) is
exactly the right scope.  It is NOT shared across multiple Streamlit processes.

Liveness: a session that dies without releasing (browser closed mid-wait, kernel
OOM, ...) must not wedge the line.  Tickets carry a ``last_seen`` heartbeat that
``try_acquire``/``heartbeat`` refresh; the reaper drops waiting tickets nobody
has polled for ``stale_waiting_s`` and frees running slots untouched for
``stale_running_s`` (a stream that produced nothing for 15 min is dead).
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional


class QueueError(Exception):
    """Base class for admission errors."""


class QueueFull(QueueError):
    """The waiting line has reached ``max_waiting``."""


class UserBusy(QueueError):
    """This user already has ``per_user_limit`` requests waiting/running."""


class QueueTimeout(QueueError):
    """The ticket waited longer than the caller's timeout."""


@dataclass
class Ticket:
    id: str
    user_id: str
    created: float          # monotonic
    last_seen: float        # monotonic; heartbeat
    started: Optional[float] = None
    finished: Optional[float] = None
    state: str = "waiting"  # waiting | running | done | cancelled | expired

    @property
    def waited_s(self) -> float:
        end = self.started if self.started is not None else time.monotonic()
        return max(0.0, end - self.created)

    @property
    def running_s(self) -> float:
        if self.started is None:
            return 0.0
        end = self.finished if self.finished is not None else time.monotonic()
        return max(0.0, end - self.started)


class RequestQueue:
    def __init__(
        self,
        max_concurrency: int = 1,
        max_waiting: int = 32,
        per_user_limit: int = 1,
        stale_waiting_s: float = 10.0,
        stale_running_s: float = 900.0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = int(max_concurrency)
        self.max_waiting = int(max_waiting)
        self.per_user_limit = int(per_user_limit)
        self.stale_waiting_s = float(stale_waiting_s)
        self.stale_running_s = float(stale_running_s)

        self._lock = threading.Lock()
        self._waiting: deque[Ticket] = deque()
        self._running: dict[str, Ticket] = {}
        # lifetime counters
        self._served = 0
        self._rejected = 0
        self._expired = 0
        self._wait_total_s = 0.0

    # ------------------------------------------------------------------ public
    def submit(self, user_id: str) -> Ticket:
        """Join the line.  Raises UserBusy / QueueFull instead of queueing."""
        with self._lock:
            now = time.monotonic()
            self._reap(now)
            active = sum(1 for t in self._waiting if t.user_id == user_id) + sum(
                1 for t in self._running.values() if t.user_id == user_id
            )
            if active >= self.per_user_limit:
                self._rejected += 1
                raise UserBusy(
                    "You already have a request in progress. Wait for it to finish "
                    "(or stop it) before sending another."
                )
            if len(self._waiting) >= self.max_waiting:
                self._rejected += 1
                raise QueueFull(
                    f"The queue is full ({self.max_waiting} waiting). Please try again in a moment."
                )
            t = Ticket(id=uuid.uuid4().hex[:12], user_id=user_id, created=now, last_seen=now)
            self._waiting.append(t)
            self._admit(now)
            return t

    def try_acquire(self, ticket: Ticket) -> bool:
        """Poll: True once this ticket holds a slot.  Also acts as a heartbeat."""
        with self._lock:
            now = time.monotonic()
            ticket.last_seen = now
            self._reap(now)
            if ticket.state == "running":
                return True
            if ticket.state != "waiting":
                raise QueueError(f"ticket {ticket.id} is {ticket.state}")
            self._admit(now)
            return ticket.state == "running"

    def heartbeat(self, ticket: Ticket) -> None:
        """Call while streaming so the reaper knows this slot is alive."""
        ticket.last_seen = time.monotonic()

    def position(self, ticket: Ticket) -> int:
        """1-based place in the waiting line; 0 once running/finished."""
        with self._lock:
            for i, t in enumerate(self._waiting):
                if t is ticket:
                    return i + 1
            return 0

    def release(self, ticket: Ticket) -> None:
        """Give the slot back (or leave the line).  Idempotent."""
        with self._lock:
            now = time.monotonic()
            if ticket.state == "running":
                self._running.pop(ticket.id, None)
                ticket.state = "done"
                ticket.finished = now
                self._served += 1
                self._wait_total_s += ticket.waited_s
            elif ticket.state == "waiting":
                try:
                    self._waiting.remove(ticket)
                except ValueError:
                    pass
                ticket.state = "cancelled"
                ticket.finished = now
            self._admit(now)

    def wait(
        self,
        ticket: Ticket,
        timeout: float,
        poll_interval: float = 0.3,
        on_tick: Optional[Callable[[int, float, dict], None]] = None,
    ) -> float:
        """Block until the ticket runs.  Returns seconds waited.

        ``on_tick(position, waited_s, stats)`` is called on every poll so the UI
        can show the line.  If it raises (e.g. Streamlit stopping the script),
        the exception propagates; the caller's ``finally`` must ``release``.
        On timeout the ticket is cancelled and QueueTimeout is raised.
        """
        deadline = time.monotonic() + timeout
        while True:
            if self.try_acquire(ticket):
                return ticket.waited_s
            if on_tick is not None:
                on_tick(self.position(ticket), ticket.waited_s, self.stats())
            if time.monotonic() >= deadline:
                self.release(ticket)
                raise QueueTimeout(
                    f"Gave up after waiting {timeout:.0f}s for a free slot. Please try again later."
                )
            time.sleep(poll_interval)

    def stats(self) -> dict:
        with self._lock:
            served = self._served
            return {
                "running": len(self._running),
                "waiting": len(self._waiting),
                "max_concurrency": self.max_concurrency,
                "max_waiting": self.max_waiting,
                "served": served,
                "rejected": self._rejected,
                "expired": self._expired,
                "avg_wait_s": (self._wait_total_s / served) if served else 0.0,
            }

    # ----------------------------------------------------------------- private
    def _admit(self, now: float) -> None:
        """Move tickets from the head of the line into free slots (caller holds lock)."""
        while self._waiting and len(self._running) < self.max_concurrency:
            head = self._waiting.popleft()
            head.state = "running"
            head.started = now
            self._running[head.id] = head

    def _reap(self, now: float) -> None:
        """Drop tickets whose owner stopped polling (caller holds lock)."""
        stale = [t for t in self._waiting if now - t.last_seen > self.stale_waiting_s]
        for t in stale:
            self._waiting.remove(t)
            t.state = "expired"
            t.finished = now
            self._expired += 1
        for tid, t in list(self._running.items()):
            if now - t.last_seen > self.stale_running_s:
                del self._running[tid]
                t.state = "expired"
                t.finished = now
                self._expired += 1
