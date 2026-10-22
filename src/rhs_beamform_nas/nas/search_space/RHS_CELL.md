# RHS Cell

src/rhs_beamforming/nas/search_space/rhs_cell.py
Conceptually:

```
                    RHS CELL
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
 Complex path      Spatial path     Config path
       │               │                │
       ▼               ▼                ▼
 Complex op          GNN            FiLM/Cross
       │               │                │
       └───────────────┼────────────────┘
                       ▼
                  Fusion node
                       │
                       ▼
                   Residual
```
This allows NAS to discover whether a particular layer benefits more from:

GNN, attention, complex residual

rather than simply choosing Conv3x3 versus Conv5x5.


## NNI Implementation

NNI's current NAS framework explicitly supports model spaces, exploration strategies and evaluators, while its one-shot methods use weight sharing to reduce the cost of searching many architectures.