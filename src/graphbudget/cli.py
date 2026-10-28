"""Reproducible patch/install/launch and whole-configuration measurement commands."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .identity import BASE_SHA, file_digest, code_digest
from .lifecycle import StartupFailure, check_port_available, http_server, startup_limit
from .runtime import write_json


def apply_patch(checkout):
    root = Path(checkout).resolve()
    patch_dir = Path(__file__).parent / "patches"
    manifest = json.loads((patch_dir / "manifest.json").read_text())
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != BASE_SHA:
        raise ValueError(f"Requires SGLang base {BASE_SHA}; got {head}")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if dirty:
        raise ValueError("Refusing to patch a dirty SGLang checkout")
    patch = patch_dir / "sglang-v0.5.9.patch"
    if file_digest(patch) != manifest["patch_sha256"]:
        raise ValueError("Packaged patch digest mismatch")
    subprocess.run(["git", "-C", str(root), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(root), "apply", str(patch)], check=True)
    for name, expected in manifest["patched_files"].items():
        if code_digest(root / "python/sglang" / name) != expected:
            raise ValueError(f"Patched file hash mismatch: {name}")
    print(json.dumps({"base_commit": head, "patch_sha256": manifest["patch_sha256"],
                      "status": "applied_and_hash_checked"}))


def _serve(args):
    """Internal owned child; public launch always supplies the HTTP supervisor."""
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory for each run")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(GRAPHBUDGET_ACTIVE="1", GRAPHBUDGET_OUTPUT=str(output),
                      GRAPHBUDGET_RUN_ID=args.run_id or uuid.uuid4().hex,
                      GRAPHBUDGET_STARTED=os.environ.get("GRAPHBUDGET_STARTED", str(time.perf_counter())),
                      GRAPHBUDGET_OBSERVE=args.observe, GRAPHBUDGET_POLICY=args.policy)
    for key in ("GRAPHBUDGET_PLAN", "GRAPHBUDGET_PROFILE", "GRAPHBUDGET_TEST_ONLY", "GRAPHBUDGET_CALIBRATE"):
        os.environ.pop(key, None)
    if args.plan:
        os.environ["GRAPHBUDGET_PLAN"] = str(Path(args.plan).resolve())
    if args.profile:
        os.environ["GRAPHBUDGET_PROFILE"] = str(Path(args.profile).resolve())
    if args.test_only:
        os.environ["GRAPHBUDGET_TEST_ONLY"] = "1"
    if args.calibrate:
        os.environ["GRAPHBUDGET_CALIBRATE"] = "1"
    argv = args.sglang_args
    if argv[:1] == ["--"]:
        argv = argv[1:]
    write_json(output / "command.json", {"python": sys.executable, "argv": sys.argv,
                                        "run_id": os.environ["GRAPHBUDGET_RUN_ID"]})
    from sglang.srt.server_args import prepare_server_args
    from sglang.launch_server import run_server
    from sglang.srt.utils import kill_process_tree
    server_args = prepare_server_args(argv)
    if server_args.grpc_mode or server_args.encoder_only:
        raise ValueError("GraphBudget v1 supervision requires the ordinary HTTP server")
    host = server_args.host
    check_port_available(host, server_args.port)
    probe_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in probe_host:
        probe_host = f"[{probe_host}]"
    write_json(output / "endpoint.json", {"run_id": os.environ["GRAPHBUDGET_RUN_ID"],
                                          "url": f"http://{probe_host}:{server_args.port}"})
    if diagnostic_path := os.environ.get("GRAPHBUDGET_REQUEST_DIAGNOSTICS"):
        from .request_diagnostics import install
        install(diagnostic_path)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory for each run")
    timeout = startup_limit(args.startup_timeout, args.plan)
    run_id = args.run_id or uuid.uuid4().hex
    command = [sys.executable, "-m", "graphbudget.cli", "_serve", "--output", str(output),
               "--run-id", run_id, "--observe", args.observe, "--policy", args.policy]
    for name in ("plan", "profile"):
        if value := getattr(args, name):
            command += ["--" + name, str(Path(value).resolve())]
    if args.test_only:
        command.append("--test-only")
    if args.calibrate:
        command.append("--calibrate")
    argv = args.sglang_args[1:] if args.sglang_args[:1] == ["--"] else args.sglang_args
    command += ["--", *argv]
    env = {**os.environ, "GRAPHBUDGET_STARTED": str(started)}
    try:
        with http_server(command, output, run_id, timeout=timeout, started=started, env=env) as (process, _):
            # Startup is complete. Runtime health management is not a watchdog:
            # a transient later /health failure must not terminate this service.
            code = process.wait()
            return code if code >= 0 else 128 - code
    except StartupFailure as error:
        print(str(error), file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        return 130


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    patch = commands.add_parser("apply-patch")
    patch.add_argument("checkout")
    for name in ("launch", "_serve"):
        server = commands.add_parser(name, help="Internal owned child" if name == "_serve" else None)
        server.add_argument("--output", required=True)
        server.add_argument("--run-id")
        server.add_argument("--plan")
        server.add_argument("--profile")
        server.add_argument("--policy", choices=("native", "cost", "test_eager"), default="native")
        server.add_argument("--observe", choices=("counts", "diagnostic"), default="counts")
        server.add_argument("--test-only", action="store_true")
        server.add_argument("--calibrate", action="store_true")
        server.add_argument("--startup-timeout", type=float, default=600,
                            help="Total HTTP-ready deadline in seconds; a plan may impose a tighter limit")
        server.add_argument("sglang_args", nargs=argparse.REMAINDER)
    measure = commands.add_parser("measure")
    measure.add_argument("experiment")
    measure.add_argument("--output", required=True)
    select = commands.add_parser("select")
    select.add_argument("measurements")
    select.add_argument("--output", required=True)
    profile = commands.add_parser("profile")
    profile.add_argument("--graph", nargs="+", required=True)
    profile.add_argument("--eager", nargs="+", required=True)
    profile.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "apply-patch":
        apply_patch(args.checkout)
    elif args.command == "launch":
        return launch(args)
    elif args.command == "_serve":
        _serve(args)
    elif args.command == "measure":
        from .experiment import measure_candidates
        measure_candidates(args.experiment, args.output)
    elif args.command == "select":
        from .plan import select_plan
        raw = json.loads(Path(args.measurements).read_text())
        write_json(args.output, select_plan(raw["identity"], raw["budget"], raw["candidates"]))
    elif args.command == "profile":
        from .experiment import build_cost_profile
        write_json(args.output, build_cost_profile(args.graph, args.eager))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
