# RHS Streaming Validation Platform

A self-contained, five-container stack for validating RHS beamforming models
against the GP reference over a framed, ACK-driven streaming protocol. It
replays held-out samples (simulator), ingests hardware captures (USB serial,
Wi-Fi/MQTT, HDMI/v4l2), persists provenance in MongoDB/Redis, and exposes a
retro black-and-white web UI.

## Services and host ports

| Service | Purpose | Host port |
| --- | --- | --- |
| `frontend` | nginx reverse proxy + retro GUI | `8088` |
| `backend` | ingest, provenance, validation, artifact browser | `8090` |
| `inference` | GP truth, DL models, JEV decisions | `8091` |
| `mongo` | samples, outputs, decisions, provenance | `27018` |
| `redis` | ACK state, streams, active model cache | `6380` |

Ports avoid the existing `rantchit` stack (80/443/8000/5173/8080).

## Quick start

```bash
cd platform
cp .env.example .env          # optional port overrides
docker compose up -d --build
docker compose ps             # all services should become healthy
```

Open <http://localhost:8088>. Health checks:

```bash
curl -s http://localhost:8088/healthz | python3 -m json.tool
curl -s http://localhost:8091/models  | python3 -m json.tool
```

## Replay the static fixture

```bash
PYTHONPATH=$PWD/platform python3 platform/firmware/simulator.py \
  --backend http://localhost:8088 --iterations 1 --verbose
```

Each iteration performs the full handshake:

```
DATA_REQ -> RECV_OK (JEV decision) -> DATA_ACK -> RESULT (diff + URLs)
```

Results are stored under a `cid` and rendered at
`/validation/<cid>`, `/samples/<cid>`, `/streams/<cid>` and
`/api/samples/<cid>`.

## Data and checkpoints

The stack mounts these read-only from the repository:

| Container path | Repo path |
| --- | --- |
| `/repo/search_results` | `search_results/` |
| `/repo/checkpoints` | `checkpoints/` |
| `/repo/src` | `src/` |
| `/repo/data/processed/mimo64_full` | `data/processed/mimo64_full/` |
| `/artifacts/*` | `search_results/`, `visualizations_review/`, `nas_comparison_results/`, `paper/` |
| `/app/fixtures` | `platform/fixtures/` |

Models:

- `hybrid_jev` — 4,083,733 params, full-data test rate R² 0.8044.
- `nas_best` — 997,825 params, NAS-selected architecture.
- `gp` — dataset GP truth used as the reference (no live GP solve).

Regenerate the four-sample fixture:

```bash
python3 platform/scripts/make_fixture.py
```

## JEV decision

`POST /jev_decide` returns the model whose beamforming output best matches the
GP truth for the current configuration:

```
score = spectral_weight * L1(probe_signature, model_signature)
        + (1 - cosine(M_dl, M_gp))
        + 0.05 * |rate_dl - rate_gp| / rate_std
        + latency_weight * normalised_latency
```

Defaults: `spectral_weight=0.2`, `latency_weight=0.05`, `k=4`. Override with
`JEV_SPECTRAL_WEIGHT`, `JEV_LATENCY_WEIGHT`, `JEV_K`.

## Hardware gateway

```bash
# USB serial (STM32)
PYTHONPATH=$PWD/platform python3 platform/firmware/simulator.py --mode serial --port /dev/ttyACM0

# Wi-Fi (ESP32 -> MQTT -> gateway)
PYTHONPATH=$PWD/platform python3 platform/firmware/simulator.py --mode wifi --broker localhost

# HDMI capture (v4l2)
PYTHONPATH=$PWD/platform python3 platform/firmware/simulator.py --mode hdmi --video-device /dev/video0
```

Firmware sources:

- `firmware/stm32/` — CRC32, frame codec, ACK state machine, JEV probe; run
  `make -C firmware/stm32 test`.
- `firmware/esp32/wifi_bridge.ino` — UART2 ↔ MQTT bridge.

## Tests

```bash
python3 -m pytest platform/tests/unit -q                 # protocol, metrics, registry
PLATFORM_URL=http://localhost:8088 \
  python3 -m pytest platform/tests/integration -q        # live stack
make -C platform/firmware/stm32 test                     # C firmware
ruff check platform
```

The integration suite skips automatically when the stack is not running.

## Known limitations

- The legacy C++/Python GP product scripts still report `rate=0.0`; the
  platform deliberately uses dataset GP `M`/`rate` as ground truth.
- Live GP solving for arbitrary hardware configs is not wired in; the probe
  requires a fixture/known configuration.
- The HDMI path captures frames but the retro GUI is the primary operator
  surface; kiosk autostart is deployment-specific.
