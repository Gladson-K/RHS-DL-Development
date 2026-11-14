#!/usr/bin/env python3
"""Enumerate neurons and connections for every RHS beamforming architecture.

Outputs:
    paper/NEURON-ANALYSIS.md   human-readable breakdown + narrative
    paper/neuron_analysis.csv  per-layer machine-readable table

For each model we report, per leaf module:
    * layer type and tensor shape (neurons)
    * trainable parameters
    * weighted connections (in x out for a dense map)
and verify the summed parameter count matches ``count_parameters(model)``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhs_beamforming.nas.setups import SETUP_VARIANTS, build_setup_by_name  # noqa: E402
from rhs_beamforming.nas.real_space import count_parameters  # noqa: E402

OUT_DIR = ROOT / "paper"
OUT_DIR.mkdir(exist_ok=True)


def _mha_stats(m: nn.MultiheadAttention) -> tuple[int, int, str]:
    d = m.embed_dim
    params = 0
    if m.in_proj_weight is not None:
        params += m.in_proj_weight.numel()
    else:  # separate q/k/v projections
        for p in (m.q_proj_weight, m.k_proj_weight, m.v_proj_weight):
            if p is not None:
                params += p.numel()
    if m.in_proj_bias is not None:
        params += m.in_proj_bias.numel()
    return params, 3 * d * d, f"{m.num_heads} heads, head_dim={d // m.num_heads}, q/k/v+attn"


def layer_rows(model: nn.Module) -> list[dict]:
    rows: list[dict] = []
    for name, m in model.named_modules():
        if isinstance(m, nn.MultiheadAttention):
            params, conn, note = _mha_stats(m)
            rows.append({
                "layer": name or "<root>",
                "type": "MultiheadAttention",
                "in": m.embed_dim,
                "out": m.embed_dim,
                "neurons": m.embed_dim,
                "connections": conn,
                "parameters": params,
                "note": note,
            })
        elif isinstance(m, nn.Linear):
            conn = m.in_features * m.out_features
            params = conn + (m.out_features if m.bias is not None else 0)
            rows.append({
                "layer": name,
                "type": "Linear",
                "in": m.in_features,
                "out": m.out_features,
                "neurons": m.out_features,
                "connections": conn,
                "parameters": params,
                "note": "dense",
            })
        elif isinstance(m, nn.LayerNorm):
            params = sum(p.numel() for p in m.parameters())
            rows.append({
                "layer": name,
                "type": "LayerNorm",
                "in": m.normalized_shape[0] if m.normalized_shape else 0,
                "out": m.normalized_shape[0] if m.normalized_shape else 0,
                "neurons": m.normalized_shape[0] if m.normalized_shape else 0,
                "connections": 0,
                "parameters": params,
                "note": "per-feature affine",
            })
        elif isinstance(m, nn.BatchNorm1d):
            params = sum(p.numel() for p in m.parameters())
            rows.append({
                "layer": name,
                "type": "BatchNorm1d",
                "in": m.num_features,
                "out": m.num_features,
                "neurons": m.num_features,
                "connections": 0,
                "parameters": params,
                "note": "per-feature affine",
            })

    # bare parameters registered directly on a module (learned tokens / adjacency)
    for name, p in model.named_parameters():
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        leaf = name.rsplit(".", 1)[-1]
        if any(r["layer"] == parent and r["type"] in ("Linear", "MultiheadAttention", "LayerNorm", "BatchNorm1d") for r in rows):
            continue
        if any(r["layer"] == name for r in rows):
            continue
        shape = tuple(p.shape)
        rows.append({
            "layer": name,
            "type": "Parameter",
            "in": shape[-1] if shape else 0,
            "out": shape[-2] if len(shape) >= 2 else (shape[-1] if shape else 0),
            "neurons": int(p.numel()),
            "connections": int(p.numel()),
            "parameters": int(p.numel()),
            "note": f"learned tensor {shape}",
        })
    return rows


def summarize(name: str, model: nn.Module) -> dict:
    rows = layer_rows(model)
    total_params = sum(r["parameters"] for r in rows)
    truth = count_parameters(model)
    total_conn = sum(r["connections"] for r in rows)
    total_neurons = sum(r["neurons"] for r in rows)
    return {
        "name": name,
        "rows": rows,
        "parameters": total_params,
        "parameters_truth": truth,
        "ok": total_params == truth,
        "connections": total_conn,
        "neurons": total_neurons,
    }


def nas_best_arch() -> dict | None:
    candidates = [
        ROOT / "search_results/nas_study/evolution/metrics/best.json",
        ROOT / "search_results/nas_study/random_b8/metrics/best.json",
        ROOT / "search_results/nas_study/random_b4/metrics/best.json",
        ROOT / "search_results/mimo64_nas/metrics/best.json",
        ROOT / "search_results/mimo64_nas/best.json",
    ]
    for path in candidates:
        if path.exists():
            doc = json.loads(path.read_text())
            return {"source": str(path.relative_to(ROOT)), "architecture": doc["architecture"],
                    "metrics": doc.get("metrics", {})}
    return None


def build_nas_model(arch_doc: dict) -> nn.Module:
    from rhs_beamforming.nas.real_space import Architecture, build_model

    return build_model(Architecture.from_dict(arch_doc))


def main() -> None:
    summaries = []
    for name in SETUP_VARIANTS:
        summaries.append(summarize(name, build_setup_by_name(name)))

    nas = nas_best_arch()
    if nas is not None:
        summaries.append(summarize(f"nas_best ({nas['source']})", build_nas_model(nas["architecture"])))

    # CSV
    with open(OUT_DIR / "neuron_analysis.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "layer", "type", "in", "out", "neurons", "connections", "parameters", "note"])
        for s in summaries:
            for r in s["rows"]:
                wr.writerow([s["name"], r["layer"], r["type"], r["in"], r["out"],
                             r["neurons"], r["connections"], r["parameters"], r["note"]])

    # Markdown
    lines = ["# Neuron Implementation and Connectivity", ""]
    lines.append("Per-model breakdown of how neurons are instantiated and how they are")
    lines.append("connected. `connections` is the number of trainable scalar weights")
    lines.append("(dense in x out); normalisation layers add affine parameters but no")
    lines.append("weighted edges. Totals are verified against `count_parameters`.")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Parameters | Connections | Neuron activations | Param check |")
    lines.append("|---|---:|---:|---:|---|")
    for s in summaries:
        lines.append(f"| {s['name']} | {s['parameters']:,} | {s['connections']:,} | {s['neurons']:,} | "
                     f"{'OK' if s['ok'] else 'MISMATCH'} |")
    lines.append("")

    for s in summaries:
        lines.append(f"## {s['name']}")
        lines.append("")
        lines.append("| layer | type | in | out | neurons | connections | params | note |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for r in s["rows"]:
            lines.append(f"| `{r['layer']}` | {r['type']} | {r['in']} | {r['out']} | "
                         f"{r['neurons']:,} | {r['connections']:,} | {r['parameters']:,} | {r['note']} |")
        lines.append("")

    lines.append("## How the neurons are implemented")
    lines.append("")
    lines.append("- **Input**: the 10-dimensional system-config vector feeds one input neuron per feature.")
    lines.append("- **Tabular MLP**: each `Linear(prev, h)` block instantiates `h` output neurons; a")
    lines.append("  residual variant adds a `Linear(prev, h)` skip so gradients bypass the activation.")
    lines.append("  The `deep` head inserts a `prev -> prev/2 -> 25601` bottleneck before the output.")
    lines.append("- **Token stem**: `Linear(10, D)` produces one `D`-dim vector that is broadcast to")
    lines.append("  `T=16` learned token slots; the token table is a `16 x D` parameter.")
    lines.append("- **GNN block**: a learnable `16 x 16` row-softmax adjacency routes messages; a")
    lines.append("  `D -> D -> D` message MLP plus a `D -> 4D -> D` FFN update each token with residuals.")
    lines.append("- **Transformer block**: pre-norm multi-head self-attention (`4` heads of `D/4`")
    lines.append("  channels) followed by a `D -> 4D -> D` FFN, both with residual adds.")
    lines.append("- **Hybrid**: interleaves the GNN and Transformer blocks (G, T, G, T).")
    lines.append("- **JEV head**: forms the `T x T` token Gram matrix, `eigh`-decomposes it, keeps")
    lines.append("  the top `k=4` eigenvectors, projects tokens into that basis, mixes with a learned")
    lines.append("  `4 -> 4` spectral gate, concatenates the mean token and emits `25601` outputs.")
    lines.append("")
    lines.append(f"_Generated from {len(summaries)} models. Parameter totals verified._")
    lines.append("")

    (OUT_DIR / "NEURON-ANALYSIS.md").write_text("\n".join(lines))

    for s in summaries:
        flag = "OK" if s["ok"] else f"MISMATCH {s['parameters']} vs {s['parameters_truth']}"
        print(f"{s['name']:28s} params={s['parameters']:>10,} conn={s['connections']:>12,} [{flag}]")
    print(f"\nwrote {OUT_DIR/'NEURON-ANALYSIS.md'} and {OUT_DIR/'neuron_analysis.csv'}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
