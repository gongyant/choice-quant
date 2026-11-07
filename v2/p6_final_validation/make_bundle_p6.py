import zipfile, hashlib, glob
from pathlib import Path
B = Path(__file__).resolve().parent; out = B / "p6_review_bundle.zip"
pats = ["*.py", "*.ps1", "*.json", "*.md", "*.csv", "results/*", "reports/*", "docs/*", "dashboard/*.py",
        "predictions_registry/*", "logs/*.log", "logs/*.jsonl", "figures/*", "tests/*"]
files = sorted({Path(f) for p in pats for f in glob.glob(str(B / p)) if Path(f).is_file() and Path(f).name not in (out.name, "p6_review_bundle.sha256.txt")})
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files: z.write(f, f.relative_to(B))
sh = hashlib.sha256(out.read_bytes()).hexdigest()
(B / "p6_review_bundle.sha256.txt").write_text(f"{sh}  p6_review_bundle.zip\n", encoding="utf-8")
print(len(files), out.stat().st_size, sh)
