# Verification & Test Plan — Endoscopic Tool Detection System

| Test ID | Requirement | Procedure | Pass Criteria | Result |
|---------|-------------|-----------|----------------|--------|
| Test-01 | REQ-01 | Run inference on held-out validation frames; record precision/recall. | mAP50 >= 0.70 | **PASS (in-domain).** YOLO11n: P 0.974, R 0.879, mAP50 0.937 on the 88-frame split; 5-fold CV mAP50 0.963 ± 0.034. Cross-dataset (m2cai16 laparoscopic, no retraining): mAP50 0.167, so the requirement holds only for the training domain. |
| Test-02 | REQ-02 | Measure end-to-end per-frame latency over 100 frames. | p95 latency < 100ms | **PARTIAL.** TensorRT engine GPU latency over 3,451 iterations: p95 0.98 ms, p99 1.03 ms, max 1.09 ms (`results/metrics/tensorrt_latency_timings.json`). End-to-end capture → UART latency not yet measured. |
| Test-03 | REQ-03 | Visually confirm GUI updates in sync with detection output over a 2-minute run. | No visible desync or crash | **NOT RUN.** GUI implemented (`src/gui/viewer.py`); soak test not recorded. |
| Test-04 | REQ-04 | Send 20 detection-state transitions; confirm receiver logs match sent count. | 20/20 messages received correctly | **NOT RUN.** Relay (`cpp/status_relay`, `src/uart_sim/serial_writer.py`) and receiver (`src/uart_sim/serial_reader.py`) implemented with `$STATE*\n` framing; transition count test not recorded. |
| Test-05 | REQ-05 | Inspect log file after a run; confirm every detection event has a timestamp entry. | 100% of events logged | **OPEN.** Status is written to a single, overwritten file; an append-only timestamped event log still has to be added. |
