import os
from pathlib import Path

ROOTS = [
    Path(r"C:\..PhD Thesis"),  # broaden to the whole thesis folder
]

NEEDLES = [
    "X_test_30.npy",
    "Combined_Features",
    "fnn_regression_model.h5",
    r"FNN_Regression",
]

EXTS = {".py", ".ipynb", ".txt", ".md", ".json", ".yaml", ".yml"}

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

matches = []
for root in ROOTS:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            t = read_text(p)
            for needle in NEEDLES:
                if needle in t:
                    matches.append((needle, str(p)))
                    break

print("Matches:")
for needle, path in matches:
    print(f"- {needle}  -->  {path}")

print(f"\nTotal matches: {len(matches)}")