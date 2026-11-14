# RHS Beamforming Streaming Validation Platform — Implementation Plan

> Status: approved plan, Phase 1 pending after the background NAS study completes.
> Scope: streaming data platform (Docker + STM32 intermediate layer), GP-vs-DL
> publication, JEV ACK-driven model switching, source→travel→destination
> validation, retro black-and-white GUI, per-result URLs, and a separate systems paper.

## Locked decisions

| Topic | Decision |
|---|---|
| HDMI | Both — capture-card input (`/dev/video*`) and display output |
| Containers | 5 services: frontend, backend, inference, MongoDB, Redis |
| Hardware | STM32F411RE Nucleo + ESP32 Wi-Fi bridge |
| Frontend | FastAPI + Jinja2 + htmx, server-rendered, deep-linkable |
| Paper | Separate systems paper under `paper-platform/` |
| Build timing | MVP starts after the background NAS study completes |

## Current state (exists vs missing)

| Area | Status |
|---|---|
| DL/GP inference API | `rhs_product/python/rhs_beamforming_product.py` (`predict_dl`, `optimize_gp`, `adaptive_beamforming`) |
| Checkpoints + data | `search_results/mimo64_full_data/checkpoints/hybrid_jev.pt`, `data/processed/mimo64_full/{X,Y,scaler,meta}` |
| Existing dashboard | `visualizations_review/html/dashboard.html` (static, to evolve) |
| Results to publish | `search_results/**`, `paper/figures/**`, `nas_study/**` |
| Docker | Daemon reachable (user in `docker` group); `docker compose 2.40.3` |
| Compose / images | `docker-compose.yaml` empty; `docker/Dockerfile.rhs-mmse-setup` is actually a Makefile (misnamed) |
| STM32 source | `/mnt/d/Workspace_Glad/DSP_On_Arm_Processors-/…/STM32F411RE` UART + CMSIS-DSP projects |
| Attached hardware | None (`/dev/ttyUSB*`, `/dev/ttyACM*`, `/dev/video*` absent) → emulate first |
| Port conflicts | 80/443/8000/5173/8080 used by `rantchit` stack → platform uses new ports |

## NAS status at planning time

`random_b4` finished (rc=0, best `h512x512x256x128`, score 1.0320). `random_b8`
started 11:47 UTC; `evolution` follows. ETA ≈ 13:20 UTC. Phase 1 uses the existing
`hybrid_jev.pt` and does not depend on the study, but implementation starts only
after the runs finish.

## Architecture

```
 Any device (STM32 / USB / HDMI / Wi-Fi)
        │
 ┌──────▼──────────────────────────────┐
 │ STM32 intermediate layer (firmware) │  frame + CRC + SEQ + ACK + JEV probe
 └──────┬───────────────┬──────────────┘
     USB-CDC         UART→ESP32 Wi-Fi        HDMI (capture / display)
        │               │                        │
 ┌──────▼───────────────▼────────────────────────▼──────┐
 │ Edge gateway (host)  pyserial / MQTT / v4l2 / watchdog │
 └──────────────────────────┬─────────────────────────────┘
                            │ HTTP/WS
 ┌──────────────────────────▼─────────────────────────────┐
 │ backend container (FastAPI): ingest, provenance,        │
 │ artifact registry, URL routes, publication job          │
 └───────┬───────────────────────────────┬────────────────┘
         │                               │
 ┌───────▼────────┐            ┌─────────▼──────────────┐
 │ Redis container│            │ MongoDB container       │
 │ streams, ACK,  │            │ gp_truth, dl_outputs,   │
 │ cache, pub/sub │            │ decisions, provenance   │
 └────────────────┘            └─────────────────────────┘
         │
 ┌───────▼──────────────────────────────────────────────┐
 │ inference/server container: GP + DL + JEV decision,   │
 │ model registry + ACK-driven model switch               │
 └───────────────────────┬───────────────────────────────┘
                         │
 ┌───────────────────────▼───────────────────────────────┐
 │ frontend container: retro B/W GUI, per-result URLs     │
 └───────────────────────────────────────────────────────┘
```

## Data contracts

**Sample record** (`SAMPLE`): `sample_id`, `cid`, 10-dim config `x`,
`gp_M` (20×64×10 complex), `gp_rate`, `source_hash`, `source_ts`, `device_id`,
`transport`.

**Frame** (serial/HDMI/Wi-Fi payload):
`SYNC|VER|TYPE|SEQ|LEN|PAYLOAD|CRC32`, where
`TYPE ∈ {DATA_REQ, RECV_OK, DATA_ACK, JEV_DECISION, MODEL_SWITCH, RESULT, ERROR}`.

**JEV probe** (STM32 → server): compact spectral signature (top-k eigenvalues of
the probe Gram / `MᴴM`) + config hash. Server returns
`RECV_OK{chosen_model, confidence, thresholds}`; STM32 replies `DATA_ACK` then
streams `RESULT`.

## JEV decision + ACK model switching

- Registry: `hybrid_jev` (DL), `nas_best` (DL, from the NAS study), `gp` (reference).
- Decision: compare probe spectrum to each model's calibration profile
  (eigenvalue distribution, historical rate/M-cos, latency budget) → choose the
  model minimizing expected error; record in `jev_decisions`.
- Static-input replay: same `x` re-sent, forcing each model in turn; verifies
  deterministic switching and beamforming-vector generation per input.
- Handshake: `DATA_REQ → RECV_OK(model) → DATA_ACK → RESULT`; Redis holds
  `ack:{seq}` and `model:active`.

## Source → Travel → Destination validation

| Stage | Captured |
|---|---|
| Source | config hash, GP M/rate, timestamp, device id, transport |
| Travel | seq, CRC pass/fail, hop, latency, loss, retransmits |
| Destination | recomputed hash, GP vs DL inference, JEV decision, deltas |

Cross-container consistency compares backend vs inference vs Mongo vs Redis per
`cid`; report `hash_match`, `rate_delta`, `M_cos`, `M_nMSE`, `e2e_latency`,
`ack_agreement`, `switch_correct`. Stored in `validation_reports`, surfaced at
`/validation/{cid}`.

## Data model

- **MongoDB (NoSQL)**: `samples`, `gp_truth`, `dl_outputs`, `jev_decisions`,
  `provenance_hops`, `model_registry`, `runs`, `artifacts`, `validation_reports`.
- **Redis**: `stream:{cid}`, `ack:{seq}`, `model:active`, `cache:result:{hash}`,
  pub/sub `events`, rate limits.

## Containers, ports, volumes

| Service | Image | Host port | Depends on |
|---|---|---|---|
| `frontend` | nginx:alpine (static) | 8088 | backend |
| `backend` | python:3.12 + FastAPI/Jinja2 | 8090 | mongo, redis, inference |
| `inference` | python:3.12 + torch | 8091 | — (mounts checkpoints/data) |
| `mongo` | mongo:7 | 27018 | — |
| `redis` | redis:7-alpine | 6380 | — |

Healthchecks, named volumes, an internal network, restart policies, and
`.env`-driven ports/secrets. MongoDB without auth inside the private network
(enable auth via `.env` if exposed).

## Repo layout (new)

```
platform/
  docker-compose.yml
  .env.example
  containers/
    backend/{Dockerfile,app/{main.py,routes.py,ingest.py,provenance.py,registry.py,models.py,templates/,static/}}
    inference/{Dockerfile,app/{main.py,registry.py,jev.py,predict.py}}
    frontend/{Dockerfile,static/}
  gateway/{bridge.py,serial_link.py,wifi_link.py,hdmi_capture.py,watchdog.py}
  firmware/stm32/{Core/,USB_DEVICE/,frame.c,crc32.c,ack.c,jev_probe.c}
  firmware/esp32/{wifi_bridge.ino}
  firmware/simulator.py
  protocol/{frames.md, schemas.py}
  tests/{unit,integration,e2e}/
paper-platform/
  RHS_Platform_IEEE.tex   (compiled via tectonic, converted via pandoc)
```

## Backend routes (retro GUI, per-result URLs)

```
/                              dashboard
/results/{run}                 run summary
/results/{run}/{artifact}      table/figure artifact
/results/{run}/figures/{name}  full-page figure
/streams/{cid}                 live stream + ACK state
/validation/{cid}              source/travel/destination diff
/models                        registry + switch history
/healthz                       health probe
```

`artifact_registry` maps `{run}/{artifact}` → file/DB record; unknown URLs return
a styled 404; browser back/forward works. Retro style is monospace 1-bit
black-and-white CSS.

## Published-data visualization

Idempotent startup ingest of `search_results/**`, `paper/figures/**`,
`nas_study/**`, plus GP-vs-DL diffs. Artifact browser + figure viewer keep the
GP = red / model = blue convention.

## Hardware layer

- **STM32**: adapt `uart.c`/`uart.h` from the DSP project; add USB CDC,
  framing/CRC, ACK state machine, JEV probe; optional CMSIS-DSP eigen helpers.
  `firmware/simulator.py` replays static samples over a PTY/MQTT so the platform
  works without a board.
- **Wi-Fi**: ESP32 bridge (MQTT/WS) via UART AT or ESP-IDF; auto-detect.
- **USB**: CDC-ACM or FTDI; `/dev/tty*` watcher.
- **HDMI**: output displays the retro GUI full-screen (`/streams/{cid}` kiosk);
  input uses a USB capture card via v4l2 (`gateway/hdmi_capture.py`).

## Tests

Unit (frame/CRC/ACK, JEV, provenance), integration (`docker compose up` +
simulator loopback), e2e (PTY USB, Mosquitto Wi-Fi, video-file HDMI), and
validation-report generation.

## Phased delivery + acceptance criteria

1. **MVP** — compose up; inference `/predict_dl`, `/predict_gp`, `/jev_decide`;
   backend ingest + artifact registry; retro UI; per-result URLs; simulator
   loopback. *Accept:* static sample round-trips, GP/DL diff stored, URLs resolve.
2. **USB hardware** — firmware + ACK/JEV switch + source→travel→destination
   report. *Accept:* switching is deterministic and logged.
3. **Wi-Fi + HDMI** — ESP32 bridge + capture-card ingest + display kiosk.
4. **Systems paper** — `paper-platform/` tex → tectonic PDF → pandoc docx.

## Risks

- No hardware attached → simulator first.
- 3.8 GB RAM → single-worker processes + inference memory cap.
- HDMI-in needs a capture card → graceful absence handling.
- MongoDB unauthenticated inside private network → enable auth via `.env` if exposed.
