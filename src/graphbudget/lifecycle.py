"""Owned POSIX server processes and one total HTTP startup deadline.

SGLang's supported runtime is Linux/WSL. Refuse Windows rather than pretend
that terminating a Windows parent also terminates all of its workers.
"""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

from .plan import load_plan
from .runtime import write_json
from .transport import HTTPTransport, peer_category


class StartupFailure(RuntimeError):
    def __init__(self, status, message, exit_code, *, child_returncode=None):
        super().__init__(message)
        self.status = status
        self.exit_code = exit_code
        self.child_returncode = child_returncode


class OwnedProcess:
    """Only processes created here may be targets of process-group cleanup."""

    def __init__(self, command, **kwargs):
        if os.name != "posix":
            raise OSError("GraphBudget server supervision requires POSIX process groups; use Linux/WSL")
        self.process = subprocess.Popen(command, start_new_session=True, **kwargs)
        self.pid = self.process.pid

    @property
    def returncode(self):
        return self.process.returncode

    def poll(self):
        return self.process.poll()

    def wait(self, timeout=None):
        return self.process.wait(timeout=timeout)


def stop_owned(process, *, grace_seconds=15):
    if process is None:
        return
    if not isinstance(process, OwnedProcess):
        raise TypeError("Refusing to clean up a process not created by OwnedProcess")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader can exit before workers, including workers ignoring TERM.
        # Always cover the remaining owned group, even after wait() succeeds.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def startup_limit(timeout, plan=None):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("startup timeout must be a finite non-negative number")
    try:
        timeout = float(timeout)
    except OverflowError as error:
        raise ValueError("startup timeout must be finite") from error
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("startup timeout must be a finite non-negative number")
    if plan is not None:
        frozen = load_plan(Path(plan).read_text(encoding="utf-8"))
        timeout = min(timeout, frozen["budget"]["max_startup_seconds"])
    return timeout


def check_port_available(host, port):
    """Reject a pre-existing listener before a health probe can accept it."""
    address = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0]
    with socket.socket(address[0], address[1], address[2]) as probe:
        # Match the HTTP server's reuse behavior across serial experiments;
        # TIME_WAIT from a finished run must not masquerade as a live listener.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(address[4])


def wait_http_ready(process, output, run_id, *, started, timeout, poll_seconds=0.05,
                    transport_mode="direct"):
    """Require the owned worker record plus pinned SGLang's warmup-gated /health.

    A successful probe must finish within the deadline; a late HTTP 200 cannot
    retroactively satisfy it. No health probing continues after this returns.
    """
    output = Path(output)
    deadline = started + timeout
    while True:
        child_code = process.poll()
        if child_code is not None:
            code = child_code if child_code > 0 else (128 - child_code if child_code < 0 else 1)
            raise StartupFailure("child_exited", f"Server exited {child_code} before HTTP readiness", code,
                                 child_returncode=child_code)
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise StartupFailure("deadline_exceeded", "HTTP-ready startup deadline exceeded", 124)
        worker_path, endpoint_path = output / "ready.json", output / "endpoint.json"
        if worker_path.exists() and endpoint_path.exists():
            try:
                worker = json.loads(worker_path.read_text(encoding="utf-8"))
                endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise StartupFailure("invalid_readiness", f"Cannot read readiness records: {error}", 126) from error
            if not isinstance(worker, dict) or not isinstance(endpoint, dict):
                raise StartupFailure("invalid_readiness", "Readiness records must be JSON objects", 126)
            if worker.get("run_id") != run_id or endpoint.get("run_id") != run_id:
                raise StartupFailure("invalid_readiness", "Readiness record belongs to another run", 126)
            if worker.get("status") != "worker_initialized":
                raise StartupFailure("invalid_readiness", "Worker did not initialize successfully", 126)
            pid = worker.get("pid")
            if type(pid) is not int or pid <= 0:
                raise StartupFailure("invalid_readiness", "Worker readiness PID is invalid", 126)
            try:
                owned = os.getpgid(pid) == process.pid
            except ProcessLookupError:
                owned = False
            if not owned:
                raise StartupFailure("invalid_readiness", "Worker PID is outside the owned process group", 126)
            url = endpoint.get("url")
            if not isinstance(url, str) or not url.startswith("http://"):
                raise StartupFailure("invalid_readiness", "Readiness endpoint URL is invalid", 126)
            try:
                transport = HTTPTransport(url, transport_mode)
                with transport.open(url + "/health", timeout=min(2, remaining)) as response:
                    healthy = response.status == 200
                    peer = peer_category(response, url)
                elapsed = time.perf_counter() - started
                if healthy and elapsed <= timeout and process.poll() is None:
                    return {"schema_version": 1, "run_id": run_id, "status": "http_ready",
                            "startup_seconds": elapsed, "startup_metric": "http_health_after_upstream_startup",
                            "startup_timeout_seconds": timeout, "server_pid": process.pid,
                            "worker_pid": pid, "url": url, "exit_code": 0,
                            "transport": transport.metadata, "transport_peer": peer}
            except (OSError, TimeoutError):
                pass
        time.sleep(min(poll_seconds, max(0, deadline - time.perf_counter())))


@contextmanager
def http_server(command, output, run_id, *, timeout, started=None, stdout=None, env=None,
                transport_mode="direct"):
    """Shared launch/wait/cleanup only; never performs benchmark requests."""
    started = time.perf_counter() if started is None else started
    process = None
    previous_term = None
    if threading.current_thread() is threading.main_thread():
        previous_term = signal.getsignal(signal.SIGTERM)
        def terminate(signum, _frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
    try:
        try:
            process = OwnedProcess(command, stdout=stdout, stderr=subprocess.STDOUT, env=env)
        except OSError as error:
            raise StartupFailure("start_failed", f"Cannot start server: {error}", 125) from error
        record = wait_http_ready(process, output, run_id, started=started, timeout=timeout,
                                transport_mode=transport_mode)
        write_json(Path(output) / "lifecycle.json", record)
        yield process, record
    except StartupFailure as error:
        write_json(Path(output) / "lifecycle.json", {
            "schema_version": 1, "run_id": run_id, "status": error.status,
            "reason": str(error), "exit_code": error.exit_code,
            "child_returncode": error.child_returncode,
            "startup_seconds": time.perf_counter() - started,
            "startup_timeout_seconds": timeout,
        })
        raise
    finally:
        try:
            stop_owned(process)
        finally:
            if previous_term is not None:
                signal.signal(signal.SIGTERM, previous_term)
