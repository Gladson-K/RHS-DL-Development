## RHS Cell

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