# Neuron Implementation and Connectivity

Per-model breakdown of how neurons are instantiated and how they are
connected. `connections` is the number of trainable scalar weights
(dense in x out); normalisation layers add affine parameters but no
weighted edges. Totals are verified against `count_parameters`.

## Summary

| Model | Parameters | Connections | Neuron activations | Param check |
|---|---:|---:|---:|---|
| mlp | 997,825 | 971,296 | 26,529 | OK |
| gnn | 3,637,761 | 3,608,448 | 30,977 | OK |
| transformer | 3,703,297 | 3,673,472 | 30,465 | OK |
| hybrid | 4,034,305 | 4,001,664 | 33,281 | OK |
| hybrid_jev | 4,083,733 | 4,050,832 | 33,541 | OK |
| nas_best (search_results/mimo64_nas/metrics/best.json) | 839,633 | 812,952 | 26,681 | OK |

## mlp

| layer | type | in | out | neurons | connections | params | note |
|---|---|---:|---:|---:|---:|---:|---|
| `net.0` | Linear | 10 | 256 | 256 | 2,560 | 2,816 | dense |
| `net.3.fc` | Linear | 256 | 256 | 256 | 65,536 | 65,792 | dense |
| `net.4.fc` | Linear | 256 | 128 | 128 | 32,768 | 32,896 | dense |
| `net.4.skip` | Linear | 256 | 128 | 128 | 32,768 | 32,896 | dense |
| `net.5.fc` | Linear | 128 | 64 | 64 | 8,192 | 8,256 | dense |
| `net.5.skip` | Linear | 128 | 64 | 64 | 8,192 | 8,256 | dense |
| `net.6` | Linear | 64 | 32 | 32 | 2,048 | 2,080 | dense |
| `net.9` | Linear | 32 | 25601 | 25,601 | 819,232 | 844,833 | dense |

## gnn

| layer | type | in | out | neurons | connections | params | note |
|---|---|---:|---:|---:|---:|---:|---|
| `stem.proj` | Linear | 10 | 128 | 128 | 1,280 | 1,408 | dense |
| `stem.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.0.msg.0` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.0.msg.2` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.0.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.0.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.0.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `blocks.0.norm2` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.1.msg.0` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.1.msg.2` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.1.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.1.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.1.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `blocks.1.norm2` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `head.0` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `head.1` | Linear | 128 | 25601 | 25,601 | 3,276,928 | 3,302,529 | dense |
| `stem.tokens` | Parameter | 128 | 16 | 2,048 | 2,048 | 2,048 | learned tensor (1, 16, 128) |
| `blocks.0.adj` | Parameter | 16 | 16 | 256 | 256 | 256 | learned tensor (16, 16) |
| `blocks.1.adj` | Parameter | 16 | 16 | 256 | 256 | 256 | learned tensor (16, 16) |

## transformer

| layer | type | in | out | neurons | connections | params | note |
|---|---|---:|---:|---:|---:|---:|---|
| `stem.proj` | Linear | 10 | 128 | 128 | 1,280 | 1,408 | dense |
| `stem.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.0.attn.attn` | MultiheadAttention | 128 | 128 | 128 | 49,152 | 49,536 | 4 heads, head_dim=32, q/k/v+attn |
| `blocks.0.attn.attn.out_proj` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.0.attn.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.0.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.0.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.0.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `blocks.1.attn.attn` | MultiheadAttention | 128 | 128 | 128 | 49,152 | 49,536 | 4 heads, head_dim=32, q/k/v+attn |
| `blocks.1.attn.attn.out_proj` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.1.attn.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.1.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.1.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.1.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `head.0` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `head.1` | Linear | 128 | 25601 | 25,601 | 3,276,928 | 3,302,529 | dense |
| `stem.tokens` | Parameter | 128 | 16 | 2,048 | 2,048 | 2,048 | learned tensor (1, 16, 128) |

## hybrid

| layer | type | in | out | neurons | connections | params | note |
|---|---|---:|---:|---:|---:|---:|---|
| `stem.proj` | Linear | 10 | 128 | 128 | 1,280 | 1,408 | dense |
| `stem.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.0.msg.0` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.0.msg.2` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.0.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.0.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.0.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `blocks.0.norm2` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.1.attn.attn` | MultiheadAttention | 128 | 128 | 128 | 49,152 | 49,536 | 4 heads, head_dim=32, q/k/v+attn |
| `blocks.1.attn.attn.out_proj` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.1.attn.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.1.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.1.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.1.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `blocks.2.msg.0` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.2.msg.2` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.2.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.2.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.2.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `blocks.2.norm2` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.3.attn.attn` | MultiheadAttention | 128 | 128 | 128 | 49,152 | 49,536 | 4 heads, head_dim=32, q/k/v+attn |
| `blocks.3.attn.attn.out_proj` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.3.attn.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.3.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.3.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.3.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `head.0` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `head.1` | Linear | 128 | 25601 | 25,601 | 3,276,928 | 3,302,529 | dense |
| `stem.tokens` | Parameter | 128 | 16 | 2,048 | 2,048 | 2,048 | learned tensor (1, 16, 128) |
| `blocks.0.adj` | Parameter | 16 | 16 | 256 | 256 | 256 | learned tensor (16, 16) |
| `blocks.2.adj` | Parameter | 16 | 16 | 256 | 256 | 256 | learned tensor (16, 16) |

## hybrid_jev

| layer | type | in | out | neurons | connections | params | note |
|---|---|---:|---:|---:|---:|---:|---|
| `stem.proj` | Linear | 10 | 128 | 128 | 1,280 | 1,408 | dense |
| `stem.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.0.msg.0` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.0.msg.2` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.0.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.0.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.0.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `blocks.0.norm2` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.1.attn.attn` | MultiheadAttention | 128 | 128 | 128 | 49,152 | 49,536 | 4 heads, head_dim=32, q/k/v+attn |
| `blocks.1.attn.attn.out_proj` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.1.attn.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.1.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.1.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.1.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `blocks.2.msg.0` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.2.msg.2` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.2.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.2.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.2.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `blocks.2.norm2` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.3.attn.attn` | MultiheadAttention | 128 | 128 | 128 | 49,152 | 49,536 | 4 heads, head_dim=32, q/k/v+attn |
| `blocks.3.attn.attn.out_proj` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `blocks.3.attn.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.3.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `blocks.3.ffn.0` | Linear | 128 | 512 | 512 | 65,536 | 66,048 | dense |
| `blocks.3.ffn.3` | Linear | 512 | 128 | 128 | 65,536 | 65,664 | dense |
| `norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `head.norm` | LayerNorm | 128 | 128 | 128 | 0 | 256 | per-feature affine |
| `head.eig_mix` | Linear | 4 | 4 | 4 | 16 | 20 | dense |
| `head.proj.0` | Linear | 256 | 128 | 128 | 32,768 | 32,896 | dense |
| `head.proj.3` | Linear | 128 | 128 | 128 | 16,384 | 16,512 | dense |
| `head.out` | Linear | 128 | 25601 | 25,601 | 3,276,928 | 3,302,529 | dense |
| `stem.tokens` | Parameter | 128 | 16 | 2,048 | 2,048 | 2,048 | learned tensor (1, 16, 128) |
| `blocks.0.adj` | Parameter | 16 | 16 | 256 | 256 | 256 | learned tensor (16, 16) |
| `blocks.2.adj` | Parameter | 16 | 16 | 256 | 256 | 256 | learned tensor (16, 16) |

## nas_best (search_results/mimo64_nas/metrics/best.json)

| layer | type | in | out | neurons | connections | params | note |
|---|---|---:|---:|---:|---:|---:|---|
| `net.0` | Linear | 10 | 384 | 384 | 3,840 | 4,224 | dense |
| `net.3.fc` | Linear | 384 | 192 | 192 | 73,728 | 73,920 | dense |
| `net.3.skip` | Linear | 384 | 192 | 192 | 73,728 | 73,920 | dense |
| `net.4.fc` | Linear | 192 | 96 | 96 | 18,432 | 18,528 | dense |
| `net.4.skip` | Linear | 192 | 96 | 96 | 18,432 | 18,528 | dense |
| `net.5.fc` | Linear | 96 | 48 | 48 | 4,608 | 4,656 | dense |
| `net.5.skip` | Linear | 96 | 48 | 48 | 4,608 | 4,656 | dense |
| `net.6` | Linear | 48 | 24 | 24 | 1,152 | 1,176 | dense |
| `net.9` | Linear | 24 | 25601 | 25,601 | 614,424 | 640,025 | dense |

## How the neurons are implemented

- **Input**: the 10-dimensional system-config vector feeds one input neuron per feature.
- **Tabular MLP**: each `Linear(prev, h)` block instantiates `h` output neurons; a
  residual variant adds a `Linear(prev, h)` skip so gradients bypass the activation.
  The `deep` head inserts a `prev -> prev/2 -> 25601` bottleneck before the output.
- **Token stem**: `Linear(10, D)` produces one `D`-dim vector that is broadcast to
  `T=16` learned token slots; the token table is a `16 x D` parameter.
- **GNN block**: a learnable `16 x 16` row-softmax adjacency routes messages; a
  `D -> D -> D` message MLP plus a `D -> 4D -> D` FFN update each token with residuals.
- **Transformer block**: pre-norm multi-head self-attention (`4` heads of `D/4`
  channels) followed by a `D -> 4D -> D` FFN, both with residual adds.
- **Hybrid**: interleaves the GNN and Transformer blocks (G, T, G, T).
- **JEV head**: forms the `T x T` token Gram matrix, `eigh`-decomposes it, keeps
  the top `k=4` eigenvectors, projects tokens into that basis, mixes with a learned
  `4 -> 4` spectral gate, concatenates the mean token and emits `25601` outputs.

_Generated from 6 models. Parameter totals verified._
