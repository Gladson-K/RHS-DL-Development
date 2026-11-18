"""Artifact registry unit tests."""

from __future__ import annotations

from pathlib import Path

from app.registry import by_run, media_type, scan


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_scan_builds_stable_urls(tmp_path: Path) -> None:
    _write(tmp_path / "search_results/mimo64_nas/best.json", "{}")
    _write(tmp_path / "search_results/mimo64_nas/plots/pareto_frontier.png")
    _write(tmp_path / "search_results/mimo64_nas/weights.pt")
    artifacts = scan((str(tmp_path / "search_results"),))
    keys = {item.key for item in artifacts}
    assert "mimo64_nas/best.json" in keys
    assert "mimo64_nas/plots/pareto_frontier.png" in keys
    assert all(not item.name.endswith(".pt") for item in artifacts)
    best = next(item for item in artifacts if item.key == "mimo64_nas/best.json")
    assert best.url == "/results/mimo64_nas/best.json"
    figure = next(item for item in artifacts if item.key.endswith("pareto_frontier.png"))
    assert media_type(figure) == "image/png"
    assert media_type(best) == "application/json"


def test_by_run_groups_artifacts(tmp_path: Path) -> None:
    _write(tmp_path / "root/run_a/x.csv")
    _write(tmp_path / "root/run_b/y.md")
    items = scan((str(tmp_path / "root"),))
    grouped = by_run(items)
    assert set(grouped) == {"run_a", "run_b"}
    assert grouped["run_a"][0].name == "x.csv"
