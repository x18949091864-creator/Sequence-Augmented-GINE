"""Report possible secrets and private paths without printing matched values."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
scan_roots = [
    root / "models/dual_branch/CrossAttn",
    root / "models/dual_branch/SequenceComparisons",
    root / "models/dual_branch/TaskModels",
    root / "experiments",
    root / "reproducibility",
    root / "results",
]
files = [root / "README.md", root / "pcqm4m_dataset.py", root / ".gitignore"] + list(root.glob("*.py"))
for folder in scan_roots:
    files += [p for p in folder.rglob("*") if p.is_file() and p.suffix in (".py",".json",".md",".txt",".csv",".sh")]
patterns = {
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "possible credential assignment": re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret|credential)\b\s*[:=]\s*['\"][^'\"]+['\"]"),
    "private absolute path": re.compile(r"/(?:home|media)/|\bubuntu3\b|\bXJW\b"),
}
issues = []
for path in files:
    if path.name in ("secret_path_check.txt", "audit_files.py"):
        continue
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeError, OSError):
        continue
    for label, pattern in patterns.items():
        if pattern.search(content):
            issues.append((str(path.relative_to(root)), label))
lines = [f"{name} | {label}" for name,label in sorted(set(issues))]
if not lines:
    lines = ["PASS no possible credential assignments, private-key blocks, or private absolute paths in retained text files."]
(root / "reproducibility/secret_path_check.txt").write_text("\n".join(lines)+"\n")
print("\n".join(lines))
