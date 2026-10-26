"""Maintainer helper: export only the reviewed patch against the exact upstream base."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphbudget.identity import BASE_SHA, code_digest, digest, file_digest

root = Path(sys.argv[1]).resolve()
destination = Path(__file__).resolve().parents[1] / "src/graphbudget/patches"
names = ["srt/server_args.py", "srt/managers/scheduler.py", "srt/model_executor/model_runner.py",
         "srt/model_executor/cuda_graph_runner.py", "srt/model_executor/forward_batch_info.py"]
head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if head != BASE_SHA:
    raise SystemExit("Wrong upstream base")
changes = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only"], text=True).splitlines()
if sorted(changes) != sorted("python/sglang/" + n for n in names):
    raise SystemExit("Unexpected changed files")
destination.mkdir(exist_ok=True)
patch = destination / "sglang-v0.5.9.patch"
patch.write_bytes(subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "HEAD"]))
package = root / "python/sglang"
tree = {p.relative_to(package).as_posix(): code_digest(p)
        for p in sorted((package / "srt").rglob("*.py"))}
manifest = {"base_commit": BASE_SHA, "patch_sha256": file_digest(patch),
            "source_hash_semantics": "SHA256 after CRLF-to-LF normalization",
            "patched_files": {n: code_digest(package / n) for n in names},
            "srt_tree_sha256": digest(tree)}
(destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
