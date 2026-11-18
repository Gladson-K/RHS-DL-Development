# STM32 intermediate layer

This directory adapts the UART/CMSIS-DSP sources from
`/mnt/d/Workspace_Glad/DSP_On_Arm_Processors-/` (STM32F411RE Nucleo) into the
streaming platform's intermediate layer.

## Files

| File | Purpose |
|---|---|
| `crc32.c/.h` | CRC-32 (IEEE 802.3) identical to Python `zlib.crc32` |
| `frame.c/.h` | `RHS1` framing: magic, version, type, seq, length, payload, CRC |
| `ack.c/.h` | ACK state machine (`DATA_REQ → RECV_OK → DATA_ACK → RESULT`) |
| `jev_probe.c/.h` | 4-bin JEV signature from a 10×10 Gram matrix (power iteration + deflation) |
| `test_frame.c` | Host-side unit test (no board needed) |
| `Makefile` | `make test` builds and runs the host test |

## Host verification

```bash
cd platform/firmware/stm32 && make test
```

Expected output: `firmware protocol tests: OK`.

## CubeIDE integration

1. Create/extend the STM32F411RE project and copy `crc32`, `frame`, `ack`,
   `jev_probe` into `Core/Src` + `Core/Inc`.
2. Reuse the UART initialization from the DSP project (`Inc/uart.h`,
   `Src/uart.c`) and redirect `printf`/logs to `USART2`.
3. Enable **USB CDC** (`USB_DEVICE/App/usbd_cdc_if.c`) for the USB transport.
4. In the main loop:
   - accumulate bytes from USB/UART,
   - call `rhs_frame_decode()`; on `RHS_FRAME_INCOMPLETE` keep buffering,
   - on `RHS_DATA_REQ`, build the JSON payload
     `{"sample_id": "...", "transport": "usb", "device_id": "stm32f411re"}`,
     send it to the ESP32 bridge (UART) or USB host,
   - feed the `RECV_OK` frame into `rhs_ack_on_frame()`, then reply `DATA_ACK`,
   - on `RESULT`, forward the beamforming vector to the DAC/RF front end.
5. For the JEV probe, form the 10×10 Gram matrix from the current beamforming
   vector (or from the received channel estimate), then call
   `rhs_jev_signature()` and include the 4 floats in the `DATA_REQ` payload as
   `"probe": [p0, p1, p2, p3]`.

## Wi-Fi (ESP32)

`../esp32/wifi_bridge.ino` bridges `Serial2` (STM32) to MQTT topics
`rhs/rx` and `rhs/tx`. Point the gateway at the same broker with
`--mode wifi`.

## Notes

- Frames are big-endian on the wire; payloads are UTF-8 JSON.
- `RHS_MAX_PAYLOAD` is 1 MiB; the F411 has 128 KiB RAM, so keep payloads small
  (sample id + config + 4-bin probe) and let the gateway attach GP truth.
- The C implementation is intentionally dependency-free so it compiles with
  `arm-none-eabi-gcc` and the host `cc` used by `make test`.
