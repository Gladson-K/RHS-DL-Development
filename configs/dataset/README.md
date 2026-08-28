## Dataset Pipeline

```
configs/dataset/v1_fixed.yaml
             │
             ▼
       Config Loader
             │
             ▼
     Parameter Sampler
             │
             ▼
      Channel Generator
             │
             ▼
        MMSE Beamformer
             │
             ▼
        RHS Excitation
             │
             ▼
       Corrected GP
             │
       ┌─────┴─────┐
       ▼           ▼
      M_GP        w_GP
       │           │
       └─────┬─────┘
             ▼
        Rate Metrics
             │
             ▼
       HDF5 + Metadata
```

## Generate Dataset


```python
python scripts/generate_dataset.py \
    --config configs/dataset/v1_fixed.yaml \
    --output data/processed/v1
```