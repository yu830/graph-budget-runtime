"""Real CPU child processes and HTTP probes; no SGLang/GPU imports required."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from unittest.mock import patch

from graphbudget import cli
from graphbudget.lifecycle import (OwnedProcess, StartupFailure, check_port_available,
                                   http_server, startup_limit, stop_owned)


SERVER = r'''
import json, os, signal, subprocess, sys, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
output, run_id, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
output.mkdir(parents=True, exist_ok=True)
def write(name, data):
    temporary = output/(name+'.tmp')
    temporary.write_text(json.dumps(data))
    temporary.replace(output/name)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if mode == 'slow_response':
            time.sleep(.6)
        self.send_response(503 if mode == 'never' or (output/'unhealthy').exists() else 200)
        self.end_headers()
    def log_message(self, *args): pass
server = HTTPServer(('127.0.0.1', 0), Handler)
write('endpoint.json', {'run_id': run_id, 'url': 'http://127.0.0.1:%s' % server.server_port})
write('ready.json', {'run_id': 'foreign' if mode == 'foreign' else run_id,
    'pid': os.getppid() if mode == 'foreign_pid' else os.getpid(), 'status': 'worker_initialized'})
if mode in ('exit', 'leader_exit', 'descendant'):
    if mode != 'exit':
        worker = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
        (output/'descendant.pid').write_text(str(worker.pid))
        time.sleep(.1)
    if mode in ('exit', 'leader_exit'):
        sys.exit(7 if mode == 'exit' else 0)
if mode == 'late': time.sleep(.6)
server.serve_forever()
'''


def process_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    # An exited grandchild can briefly remain a zombie until init reaps it.
    return not stat.exists() or stat.read_text().split()[2] != "Z"


class StartupContractTests(unittest.TestCase):
    def test_deadline_validation_and_frozen_limit(self):
        for value in (True, None, [], -1, float("nan"), float("inf"), 10**400):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                startup_limit(value)
        with patch("graphbudget.lifecycle.load_plan", return_value={"budget": {"max_startup_seconds": 3}}), \
                tempfile.TemporaryDirectory() as td:
            plan = Path(td) / "plan.json"
            plan.write_text("{}")
            self.assertEqual(startup_limit(8, plan), 3)
            self.assertEqual(startup_limit(2, plan), 2)

    def test_cli_propagates_supervisor_exit_code(self):
        with patch("graphbudget.cli.launch", return_value=124):
            self.assertEqual(cli.main(["launch", "--output", "unused"]), 124)

    def test_refuses_foreign_process_handle(self):
        with self.assertRaisesRegex(TypeError, "not created"):
            stop_owned(object())

    def test_non_posix_refuses_before_spawning_a_child(self):
        with patch("graphbudget.lifecycle.os.name", "nt"), \
                patch("graphbudget.lifecycle.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(OSError, "POSIX"):
                OwnedProcess([sys.executable, "-c", "raise SystemExit(0)"])
            spawn.assert_not_called()


@unittest.skipUnless(os.name == "posix", "Owned worker process groups require Linux/WSL")
class ProcessLifecycleTests(unittest.TestCase):
    def command(self, output, mode):
        return [sys.executable, "-u", "-c", SERVER, str(output), "cpu", mode]

    def cli_fixture(self, root, mode):
        """A CPU-only stand-in exercises the real public launch subprocess."""
        sources = root / "fixtures"
        package = sources / "sglang"
        (package / "srt").mkdir(parents=True)
        (package / "__init__.py").touch()
        (package / "srt/__init__.py").touch()
        (package / "srt/server_args.py").write_text(
            "from types import SimpleNamespace\n"
            "def prepare_server_args(argv):\n"
            "    return SimpleNamespace(host='127.0.0.1', port=0, grpc_mode=False, encoder_only=False)\n")
        (package / "srt/utils.py").write_text("def kill_process_tree(*args, **kwargs): pass\n")
        (package / "launch_server.py").write_text(
            "import os, sys\n"
            "def run_server(args):\n"
            "    sys.argv = ['cpu', os.environ['GRAPHBUDGET_OUTPUT'], os.environ['GRAPHBUDGET_RUN_ID'], "
            + repr(mode) + "]\n    exec(" + repr(SERVER) + ", {'__name__': '__main__'})\n")
        source_root = str(Path(__file__).resolve().parents[1] / "src")
        return {**os.environ, "PYTHONPATH": os.pathsep.join((str(sources), source_root))}

    def test_public_launch_enforces_http_deadline_and_propagates_exit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "runtime"
            result = subprocess.run([sys.executable, "-m", "graphbudget.cli", "launch", "--output",
                                     str(output), "--startup-timeout", ".5", "--run-id", "cpu"],
                                    env=self.cli_fixture(root, "late"), capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 124, result.stderr.decode())
            self.assertEqual(json.loads((output / "lifecycle.json").read_text())["status"], "deadline_exceeded")
            worker = json.loads((output / "ready.json").read_text())
            self.assertFalse(process_running(worker["pid"]))

    def test_public_launch_sigterm_cleans_owned_server(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "runtime"
            process = subprocess.Popen([sys.executable, "-m", "graphbudget.cli", "launch", "--output",
                                        str(output), "--startup-timeout", "3", "--run-id", "cpu"],
                                       env=self.cli_fixture(root, "on_time"), stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            try:
                end = time.monotonic() + 5
                while not (output / "lifecycle.json").exists() and process.poll() is None and time.monotonic() < end:
                    time.sleep(.02)
                record = json.loads((output / "lifecycle.json").read_text())
                self.assertEqual(record["status"], "http_ready")
                process.terminate()
                self.assertEqual(process.wait(timeout=5), 143)
                self.assertFalse(process_running(record["server_pid"]))
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)

    def test_worker_done_but_late_or_never_http_ready(self):
        for mode in ("late", "never", "slow_response"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                output = Path(td)
                start = time.perf_counter()
                with self.assertRaises(StartupFailure) as raised:
                    with http_server(self.command(output, mode), output, "cpu", timeout=.3):
                        self.fail("late HTTP cannot pass")
                self.assertEqual(raised.exception.status, "deadline_exceeded")
                self.assertEqual(raised.exception.exit_code, 124)
                self.assertLess(time.perf_counter() - start, 2)
                worker = json.loads((output / "ready.json").read_text())
                self.assertEqual(worker["status"], "worker_initialized")
                self.assertFalse(process_running(worker["pid"]))
                self.assertEqual(json.loads((output / "lifecycle.json").read_text())["status"], "deadline_exceeded")

    def test_on_time_and_transient_failure_does_not_kill_service(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            with http_server(self.command(output, "on_time"), output, "cpu", timeout=3) as (process, ready):
                self.assertEqual(ready["status"], "http_ready")
                (output / "unhealthy").touch()
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(ready["url"] + "/health", timeout=1)
                time.sleep(.1)
                self.assertIsNone(process.poll())
            self.assertFalse(process_running(process.pid))

    def test_early_exit_and_start_failure_have_explicit_codes(self):
        with tempfile.TemporaryDirectory() as td:
            for command, status, code in ((self.command(td, "exit"), "child_exited", 7),
                                           ([str(Path(td) / "missing-python")], "start_failed", 125)):
                with self.subTest(status=status), self.assertRaises(StartupFailure) as raised:
                    with http_server(command, td, "cpu", timeout=3):
                        self.fail("must not start")
                self.assertEqual((raised.exception.status, raised.exception.exit_code), (status, code))

    def test_foreign_run_and_port_conflict_are_not_readiness(self):
        for mode, message in (("foreign", "another run"), ("foreign_pid", "outside the owned")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                with self.assertRaisesRegex(StartupFailure, message):
                    with http_server(self.command(td, mode), td, "cpu", timeout=3):
                        self.fail("must not start")
        with tempfile.TemporaryDirectory() as td:
            with http_server(self.command(td, "on_time"), td, "cpu", timeout=3) as (_, ready):
                with self.assertRaises(OSError):
                    check_port_available("127.0.0.1", int(ready["url"].rsplit(":", 1)[1]))

    def test_cleanup_kills_descendants_after_leader_exit_but_preserves_unrelated(self):
        unrelated = OwnedProcess([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            for mode in ("descendant", "leader_exit"):
                with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                    output = Path(td)
                    process = OwnedProcess(self.command(output, mode))
                    try:
                        end = time.monotonic() + 3
                        while not (output / "descendant.pid").exists() and time.monotonic() < end:
                            time.sleep(.01)
                        child_pid = int((output / "descendant.pid").read_text())
                        time.sleep(.2)
                        stop_owned(process, grace_seconds=.2)
                        end = time.monotonic() + 2
                        while process_running(child_pid) and time.monotonic() < end:
                            time.sleep(.01)
                        self.assertFalse(process_running(child_pid))
                        self.assertIsNone(unrelated.poll())
                    finally:
                        stop_owned(process, grace_seconds=.2)
        finally:
            stop_owned(unrelated, grace_seconds=.2)


if __name__ == "__main__":
    unittest.main()
