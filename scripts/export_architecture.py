import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search-result", required=True)
    p.add_argument("--output", default="configs/model/nas_best.json")
    args = p.parse_args()

    source = Path(args.search_result)
    if source.is_dir():
        source = source / "candidates.json"

    candidates = json.loads(source.read_text(encoding="utf-8"))
    best = max(candidates, key=lambda x: x["score"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Exported: {out}")


if __name__ == "__main__":
    main()
