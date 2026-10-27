"""Rebuild the exact packaged patch/manifest from this task's upstream worktree."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from graphbudget.identity import BASE_SHA, code_digest, digest, file_digest

source=Path(sys.argv[1]).resolve()
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==BASE_SHA
patch=ROOT/'src/graphbudget/patches/sglang-v0.5.9.patch'
patch.write_bytes(subprocess.check_output(['git','-C',str(source),'diff','--binary','HEAD']))
names=subprocess.check_output(['git','-C',str(source),'diff','--name-only','HEAD'],text=True).splitlines()
assert all(n.startswith('python/sglang/srt/') for n in names)
sg=source/'python/sglang'
tree={str(p.relative_to(sg)).replace('\\','/'):code_digest(p) for p in sorted((sg/'srt').rglob('*.py'))}
manifest={'base_commit':BASE_SHA,'patch_sha256':file_digest(patch),
          'source_hash_semantics':'SHA256 after CRLF-to-LF normalization',
          'patched_files':{n[len('python/sglang/'):]:code_digest(source/n) for n in names},
          'srt_tree_sha256':digest(tree)}
(patch.parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
