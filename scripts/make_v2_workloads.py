"""Generate the fixed v2 interactive and previously unused acceptance inputs."""
import argparse
import json
from pathlib import Path

from graphbudget.benchmark import load_workload
from graphbudget.identity import file_digest
from graphbudget.runtime import write_json

ROOT = Path(__file__).resolve().parents[1]


def make_workloads(source):
    original = json.loads(Path(source).read_text())
    rows = original["requests"]
    if len(rows) < 43:
        raise ValueError("The fixed calibration source needs at least 43 requests")
    shared = {
        "schema_version": 1,
        "kind": "graphbudget_workload",
        "token_ids_validated_against_model": True,
        "token_ids_note": "Concatenations of already published Qwen2.5 tokenizer IDs.",
    }
    interactive = {
        **shared,
        "workload_id": "decode-v2-interactive-8x64",
        "purpose": "interactive_regression",
        "description": "Eight spaced requests; run with one client worker.",
        "requests": [
            {"request_id": f"interactive-{i:02}",
             "input_ids": rows[(5 * i + 3) % len(rows)]["input_ids"],
             "planned_at_s": i * 1.25, "max_new_tokens": 64}
            for i in range(8)
        ],
    }
    held = []
    for wave, (count, at) in enumerate(((2, 0.0), (4, 1.2), (6, 3.0), (12, 5.5))):
        for index in range(count):
            i = len(held)
            left = rows[(7 * i + 5) % len(rows)]["input_ids"]
            right = rows[(11 * i + 1) % len(rows)]["input_ids"]
            held.append({"request_id": f"acceptance-{wave}-{index:02}",
                         "input_ids": left + right + left[:2],
                         "planned_at_s": at,
                         "max_new_tokens": (32, 64, 128)[i % 3]})
    acceptance = {
        **shared,
        "workload_id": "decode-v2-acceptance-24-mixed-bursts",
        "purpose": "heldout_acceptance",
        "description": "New concatenated prompts and four mixed-length arrival bursts.",
        "requests": held,
    }
    return {"interactive": load_workload(interactive).to_dict(),
            "acceptance": load_workload(acceptance).to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "examples/workloads/calibration.json"
    manifest = {"calibration_source_sha256": file_digest(source), "files": {}}
    for name, workload in make_workloads(source).items():
        path = out / f"{name}.json"
        write_json(path, workload)
        manifest["files"][name] = {"sha256": file_digest(path), "requests": len(workload["requests"])}
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
