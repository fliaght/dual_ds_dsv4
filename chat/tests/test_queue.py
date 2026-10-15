import os, sys, threading, time, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from request_queue import QueueFull, QueueTimeout, RequestQueue, UserBusy, QueueError


class QueueTests(unittest.TestCase):
    def test_fifo_and_concurrency_limit(self):
        q = RequestQueue(max_concurrency=1, max_waiting=10)
        order, running_max, lock = [], [0], threading.Lock()
        n_running = [0]

        def worker(uid, delay):
            t = q.submit(uid)
            q.wait(t, timeout=10)
            with lock:
                n_running[0] += 1
                running_max[0] = max(running_max[0], n_running[0])
                order.append(uid)
            time.sleep(delay)
            with lock:
                n_running[0] -= 1
            q.release(t)

        threads = []
        for i in range(5):
            th = threading.Thread(target=worker, args=(f"u{i}", 0.05))
            threads.append(th); th.start(); time.sleep(0.01)   # deterministic arrival order
        for th in threads: th.join()
        self.assertEqual(order, [f"u{i}" for i in range(5)])
        self.assertEqual(running_max[0], 1)
        s = q.stats()
        self.assertEqual((s["running"], s["waiting"], s["served"]), (0, 0, 5))

    def test_concurrency_two(self):
        q = RequestQueue(max_concurrency=2, max_waiting=10)
        peak, cur, lock = [0], [0], threading.Lock()

        def worker(uid):
            t = q.submit(uid); q.wait(t, timeout=10)
            with lock:
                cur[0] += 1; peak[0] = max(peak[0], cur[0])
            time.sleep(0.05)
            with lock: cur[0] -= 1
            q.release(t)
        ths = [threading.Thread(target=worker, args=(f"u{i}",)) for i in range(6)]
        for th in ths: th.start()
        for th in ths: th.join()
        self.assertEqual(peak[0], 2)

    def test_position_and_per_user_limit(self):
        q = RequestQueue(max_concurrency=1, max_waiting=10)
        a = q.submit("alice"); self.assertTrue(q.try_acquire(a))       # runs immediately
        b = q.submit("bob");   self.assertFalse(q.try_acquire(b)); self.assertEqual(q.position(b), 1)
        c = q.submit("carol"); self.assertEqual(q.position(c), 2)
        with self.assertRaises(UserBusy):
            q.submit("bob")                                                # one in-flight per user
        q.release(a)
        self.assertTrue(q.try_acquire(b)); self.assertEqual(q.position(c), 1)
        q.release(b); q.release(c); q.release(c)                           # idempotent
        self.assertEqual(q.stats()["waiting"], 0)

    def test_queue_full_and_timeout(self):
        q = RequestQueue(max_concurrency=1, max_waiting=1)
        a = q.submit("a"); q.try_acquire(a)
        b = q.submit("b")
        with self.assertRaises(QueueFull):
            q.submit("c")
        t0 = time.monotonic()
        with self.assertRaises(QueueTimeout):
            q.wait(b, timeout=0.3, poll_interval=0.05)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(b.state, "cancelled")
        self.assertEqual(q.stats()["waiting"], 0)
        with self.assertRaises(QueueError):
            q.try_acquire(b)

    def test_stale_tickets_are_reaped(self):
        q = RequestQueue(max_concurrency=1, max_waiting=10, stale_waiting_s=0.1, stale_running_s=0.2)
        a = q.submit("a"); q.try_acquire(a)          # running, then its session "dies"
        b = q.submit("b")                            # waiting, never polls again
        time.sleep(0.15)
        c = q.submit("c")                            # triggers reap of b (stale waiting)
        self.assertEqual(b.state, "expired")
        time.sleep(0.15)
        self.assertTrue(q.try_acquire(c))            # a expired (stale running) -> c admitted
        self.assertEqual(a.state, "expired")
        q.release(c)
        self.assertEqual(q.stats()["expired"], 2)

    def test_on_tick_exception_propagates_and_release_cleans(self):
        q = RequestQueue(max_concurrency=1)
        a = q.submit("a"); q.try_acquire(a)
        b = q.submit("b")
        class Stop(BaseException): pass
        def tick(pos, waited, stats): raise Stop()
        with self.assertRaises(Stop):
            q.wait(b, timeout=5, on_tick=tick)
        q.release(b)                                  # what the app's finally does
        self.assertEqual(q.stats()["waiting"], 0)
        q.release(a)


if __name__ == "__main__":
    unittest.main()
