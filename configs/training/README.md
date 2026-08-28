## Training Pipeline


```python
python scripts/train.py \
    --dataset data/processed/v1 \
    --model configs/model/holobeam_gen_v1.yaml \
    --training configs/training/physics_aware.yaml
```

```
HDF5
 │
 ▼
Dataset
 │
 ▼
Complex encoder
 │
 ▼
HoloBeam-GEN
 │
 ├── M prediction
 ├── w prediction
 └── rate prediction
 │
 ▼
Loss
 │
 ├── beamforming
 ├── rate
 ├── power
 └── phase
 │
 ▼
Optimizer
 │
 ▼
Checkpoint
```