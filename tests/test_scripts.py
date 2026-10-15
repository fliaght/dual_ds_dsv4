"""Run shell entry points with isolated config and command substitutes, never GPUs."""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dsv4-check-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "project with spaces"
        self.root.mkdir()
        for name in ("scripts", "config", "bench"):
            shutil.copytree(ROOT / name, self.root / name, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
        self.log = self.root / "logs"
        config = (ROOT / "cluster.conf.example").read_text()
        config += f"\nLOG_DIR={shlex.quote(str(self.log))}\nREMOTE_STAGE=/tmp/mock-worker\n"
        (self.root / "cluster.conf").write_text(config)
        self.bin = self.root / "fake-bin"
        self.bin.mkdir()
        self.calls = self.root / "calls.jsonl"
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        MOCK_CALLS=str(self.calls))
        self.env.pop("RECIPE", None)
        self.env.pop("BENCH_PYTHON", None)

    def fake(self, name, body):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
                        "with open(os.environ['MOCK_CALLS'], 'a') as f:\n"
                        "    f.write(json.dumps(sys.argv) + '\\n')\n" + body)
        path.chmod(0o755)

    def run_shell(self, code, **env):
        return subprocess.run(["bash", "-euc", code], cwd=self.root,
                              env={**self.env, **env}, capture_output=True, text=True, timeout=15)

    def read_calls(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_optional_config_and_recipe_alias_preserve_values(self):
        (self.root / "cluster.conf").unlink()
        for recipe in ("interactive-128k", "interactive-200k"):
            result = self.run_shell('source scripts/_config.sh; load_cluster_config optional; '
                                    'echo "$MAX_MODEL_LEN $MAX_NUM_SEQS $GPU_MEM_UTIL"', RECIPE=recipe)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "131072 1 0.80")
        self.assertFalse(self.log.exists(), "configuration loading must not create runtime directories")

    def test_missing_config_and_unknown_recipe_fail_early(self):
        result = self.run_shell('source scripts/_config.sh; load_cluster_config', RECIPE="../escape")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid recipe", result.stderr)
        result = self.run_shell('source scripts/_config.sh; load_cluster_config', RECIPE="missing")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found", result.stderr)
        (self.root / "cluster.conf").unlink()
        result = self.run_shell('source scripts/_config.sh; load_cluster_config')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cluster.conf missing", result.stderr)

    def test_environment_recipe_override(self):
        result = self.run_shell('source scripts/_config.sh; load_cluster_config; '
                                'echo "$MAX_MODEL_LEN $MAX_NUM_SEQS"', RECIPE="concurrent-32k")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "32768 4")

    def test_benchmark_entrypoint_uses_bench_directory(self):
        self.fake("python3", "print('offline benchmark substitute')\n")
        result = self.run_shell("bash bench/run_full_bench.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.read_calls()
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(Path(call[1]), self.root / "bench/bench_openai.py")
            self.assertTrue(Path(call[1]).is_file())

    def test_worker_sync_includes_transitive_launcher_dependencies(self):
        self.fake("scp", "")
        result = self.run_shell('source scripts/_lib.sh; _is_local() { return 1; }; '
                                'ssh_node() { :; }; push_node 192.0.2.2')
        self.assertEqual(result.returncode, 0, result.stderr)
        sources = [Path(arg).name for call in self.read_calls() for arg in call[1:-1] if arg != "-q"]
        self.assertCountEqual(sources, ["cluster.conf", "run-node.sh", "_lib.sh", "_config.sh",
                                       "nccl-env.sh", "interactive-128k.env"])

    def test_rank_launcher_keeps_existing_engine_configuration(self):
        self.fake("docker", "print('mock-container')\n")
        result = self.run_shell("NODE_RANK=1 bash scripts/run-node.sh", RECIPE="interactive-200k")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.read_calls()
        run = next(call for call in calls if call[1] == "run")
        inner = run[-1]
        for flag in ("--headless", "--max-model-len 131072", "--tensor-parallel-size 2",
                     "--max-num-seqs 1", "--gpu-memory-utilization 0.80", "--reasoning-parser deepseek_v4"):
            self.assertIn(flag, inner)
        self.assertIn('MTP_CONF={"method":"mtp","num_speculative_tokens":2}', run)

    def test_v41_preflight_missing_assets_never_calls_docker(self):
        with (self.root / 'cluster.conf').open('a') as config:
            config.write(f'\nV41_MODEL_DIR={shlex.quote(str(self.root / "missing-model"))}\n'
                         f'V41_ENGRAM_DIR={shlex.quote(str(self.root / "missing-engram"))}\n')
        self.fake('docker', 'raise SystemExit("must not launch")\n')
        result = self.run_shell('bash scripts/v41-service.sh local-precheck', RECIPE='v41-256k')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls.exists(), 'incomplete assets must fail before touching Docker')

    def test_v41_preflight_rejects_low_free_despite_high_available(self):
        self.fake('python3', "print(json.dumps({'conservative_rank_mib': 102824}) "
                  "if 'v41_assets.py' in ' '.join(sys.argv) else '102824')\n")
        self.fake('awk', "print(125000 if 'MemAvailable' in ' '.join(sys.argv) else 50000)\n")
        self.fake('docker', 'raise SystemExit("must not launch")\n')
        result = self.run_shell('bash scripts/v41-service.sh local-precheck', RECIPE='v41-256k')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Insufficient physical free pages', result.stderr)
        self.assertFalse(any(Path(call[0]).name == 'docker' for call in self.read_calls()))

    def test_v41_node_refuses_to_allocate_without_guard(self):
        self.fake('docker', 'raise SystemExit("must not launch")\n')
        result = self.run_shell('NODE_RANK=0 bash scripts/v41-node.sh', RECIPE='v41-256k')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls.exists(), 'missing guard must prevent all GPU allocation')

    def test_v41_mismatched_rank_settings_fail_before_docker(self):
        with (self.root / 'cluster.conf').open('a') as config:
            config.write('\nNODES="192.0.2.1 192.0.2.2"\n')
        self.fake('ip', "print('eth0 UP 192.0.2.1/24')\n")
        self.fake('ssh', "if 'local-settings' in ' '.join(sys.argv): print('worker-different-settings')\n")
        self.fake('scp', 'pass\n')
        self.fake('docker', 'raise SystemExit("must not launch")\n')
        result = self.run_shell('bash scripts/v41-service.sh start', RECIPE='v41-256k',
                                V41_SPEC_METHOD='dspark')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Inference settings differ between nodes', result.stderr)
        self.assertFalse(any(Path(call[0]).name == 'docker' for call in self.read_calls()))

    def smoke(self, replies):
        fixture = self.root / "responses.json"
        fixture.write_text(json.dumps(replies))
        self.fake("curl", """records = json.loads(Path(os.environ['MOCK_RESPONSES']).read_text())
index = len(Path(os.environ['MOCK_CALLS']).read_text().splitlines()) - 1
reply = records[index]
if reply == 'HTTP_ERROR':
    raise SystemExit(22)
print(json.dumps(reply))
""")
        return self.run_shell("bash bench/smoke.sh", MOCK_RESPONSES=str(fixture))

    @staticmethod
    def answer(text):
        return {"choices": [{"message": {"content": text}}], "usage": {"completion_tokens": 3}}

    def test_smoke_uses_exactly_five_requests(self):
        result = self.smoke([self.answer("391"), *[self.answer("OK") for _ in range(4)]])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.read_calls()), 5)

    def test_smoke_rejects_api_errors_and_missing_answers(self):
        result = self.smoke([self.answer("391"), self.answer("OK"),
                             {"error": {"message": "busy"}}, "HTTP_ERROR", self.answer("")])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL", result.stdout)

    def test_smoke_rejects_wrong_math_even_with_five_answers(self):
        result = self.smoke([self.answer("3910"), *[self.answer("OK") for _ in range(4)]])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.read_calls()), 5)

    def test_tf_setup_drops_lossy_keys_and_never_overwrites_env(self):
        tf = self.root / "tf"
        (tf / ".git").mkdir(parents=True)
        (tf / "patches").mkdir()
        (tf / "config").mkdir()
        (tf / "config" / "prod.env.example").write_text(
            'WORKER_SSH="<user>@<worker>"\nNCCL_IB_HCA=rocep1s0f1\nTF_DSV41_EXPERT_TOPP=0.85\n'
            'TF_DSV41_EXPERT_TOPP_LIKE=1\nCONTEXT=300000\nPARALLEL=4\n')
        with (self.root / "cluster.conf").open("a") as config:
            config.write(f'\nNODES="192.0.2.1 192.0.2.2"\nV41_PACKED_DIR={shlex.quote(str(self.root / "packed"))}\n')
        self.fake("ip", "print('eth0 UP 198.51.100.9/24')\n")    # neither node local: both via ssh
        self.fake("ssh", "print('rocep1s0f0,roceP2p1s0f0,', end='')\n")
        self.fake("git", "print('2dcce0303451da2a59f80e6c7413da8b3b0f7a57' if 'rev-parse' in sys.argv else '')\n")
        env = dict(RECIPE="v41-416k", TF_DIR=str(tf), TF_STATE=str(self.root / "tf-state"))
        result = self.run_shell("bash scripts/tf-service.sh setup", **env)
        self.assertEqual(result.returncode, 0, result.stderr)
        prod = tf / "config" / "prod.env"
        effective = self.run_shell(f"set -a; source {shlex.quote(str(prod))}; set +a; env", **env).stdout
        for line in ("CONTEXT=425984", "PARALLEL=1", "TF_DSV41_EXPERT_TOPP_LIKE=1", "WORKER_SSH=",
                     "HEAD_IB_HCA=rocep1s0f0,roceP2p1s0f0", "TF_DSV41_PREFILL=full", "PORT=8100\n",
                     "SERVED_NAME=dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"):
            self.assertIn(line, effective)
        self.assertNotIn("NCCL_IB_HCA=", effective)
        self.assertNotIn("TF_DSV41_EXPERT_TOPP=", effective)
        self.assertIn("@192.0.2.2", effective)
        self.assertIn("is current", self.run_shell("bash scripts/tf-service.sh setup", **env).stdout)
        prod.write_text(prod.read_text() + "LOCAL_EDIT=1\n")
        self.run_shell("bash scripts/tf-service.sh setup", **env)
        self.assertIn("LOCAL_EDIT=1", prod.read_text())               # never overwritten
        self.assertTrue((tf / "config" / "prod.env.new").exists())

    def test_tf_gateway_restart_waits_for_idle_and_points_at_tensorfold(self):
        with (self.root / "cluster.conf").open("a") as config:
            config.write(f'\nNODES="192.0.2.1 192.0.2.2"\nV41_PACKED_DIR={shlex.quote(str(self.root / "packed"))}\n')
        self.fake("ip", "print('eth0 UP 192.0.2.1/24')\n")
        self.fake("curl", "print(os.environ.get('MOCK_STATUS', '{\"busy\": false, \"queued\": 0}'))\n")
        self.fake("systemctl", "")
        self.fake("systemd-run", "")
        busy = self.run_shell("bash scripts/tf-service.sh gateway", RECIPE="v41-416k",
                              MOCK_STATUS='{"busy": true, "queued": 0}')
        self.assertEqual(busy.returncode, 1)
        self.assertIn("serving a request", busy.stderr)
        self.assertFalse(any(call[0].endswith("systemd-run") for call in self.read_calls()))
        result = self.run_shell("bash scripts/tf-service.sh gateway", RECIPE="v41-416k", TF_GATEWAY_HOST="0.0.0.0")
        self.assertEqual(result.returncode, 0, result.stderr)
        run = next(call for call in self.read_calls() if call[0].endswith("systemd-run"))
        self.assertIn("--unit=dsv41-tf-gateway", run)
        self.assertIn("MemoryMax=256M", run)
        args = run[run.index("/usr/bin/python3") + 1:]
        self.assertTrue(args[0].endswith("scripts/tf_gateway.py"))
        self.assertEqual(args[1:7], ["--host", "0.0.0.0", "--port", "8000", "--upstream", "http://127.0.0.1:8100"])
        probes = [call[-1] for call in self.read_calls() if call[0].endswith("curl")]
        self.assertTrue(probes and all(url == "http://127.0.0.1:8000/gateway/status" for url in probes))


if __name__ == "__main__":
    unittest.main()
