# Requirements — Endoscopic Tool Detection System

| ID | Requirement | Rationale | Verification Method |
|----|-------------|-----------|----------------------|
| REQ-01 | The system shall detect surgical tools in endoscopic video frames with confidence >= 0.4. | Core detection function. | Test-01 |
| REQ-02 | The system shall process each frame and report detection status within 100ms on target hardware. | Real-time operator feedback. | Test-02 |
| REQ-03 | The system shall display the live video feed with detection overlays in a GUI panel. | Operator situational awareness. | Test-03 |
| REQ-04 | The system shall report tool-detection status over a serial (UART) interface using a defined message format. | Interoperability with companion embedded hardware/display. | Test-04 |
| REQ-05 | The system shall log all detection events with timestamp for post-procedure review. | Traceability, consistent with regulated device recordkeeping practice. | Test-05 |

Verification results for each requirement are recorded in [test_plan.md](test_plan.md).
