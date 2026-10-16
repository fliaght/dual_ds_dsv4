"""Runs app_offline_cases.py (the page against a fake model server) in its own process, so its settings,
queue and client never leak into the other tests in this run."""
import subprocess, sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class AppOffline(unittest.TestCase):
    def test_page_flows(self):
        out = subprocess.run([sys.executable, "-m", "unittest", "app_offline_cases"], cwd=HERE,
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stdout[-2000:] + out.stderr[-6000:])
        self.assertIn("OK", out.stderr)


if __name__ == "__main__":
    unittest.main()
