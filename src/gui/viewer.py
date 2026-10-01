"""
Minimal Qt (PySide6) viewer styled like a device control panel.
Runs YOLO inference itself in Python for simplicity/speed (the C++ app in
infer_cpp/ is your separate embedded/C++ evidence — they don't need to be
fused into one binary for a same-day demo).

Usage:
    python viewer.py <frames_dir> <model.pt> [status_file]
"""
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from ultralytics import YOLO


class Viewer(QMainWindow):
    def __init__(self, frames_dir: Path, model_path: str, status_file: str):
        super().__init__()
        self.setWindowTitle("Endoscopic Tool Detection — Live Panel")
        self.resize(900, 700)

        self.model = YOLO(model_path)
        self.frames = sorted(
            [p for p in frames_dir.glob("*.jpg")] + [p for p in frames_dir.glob("*.png")]
        )
        if not self.frames:
            raise RuntimeError(f"No frames found in {frames_dir}")
        self.idx = 0
        self.status_file = status_file

        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.status_label = QLabel("STATUS: —")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; padding: 12px; "
            "background-color: #222; color: #0f0; border-radius: 6px;"
        )

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(80)  # ~12 fps, plenty for a demo

    def next_frame(self):
        path = self.frames[self.idx % len(self.frames)]
        self.idx += 1

        frame = cv2.imread(str(path))
        results = self.model.predict(frame, conf=0.4, verbose=False)[0]
        annotated = results.plot()

        detected = len(results.boxes) > 0
        self.set_status(detected)

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(800, 600, Qt.KeepAspectRatio)
        )

    def set_status(self, detected: bool):
        text = "TOOL DETECTED" if detected else "CLEAR"
        color = "#0f0" if detected else "#888"
        self.status_label.setText(f"STATUS: {text}")
        self.status_label.setStyleSheet(
            f"font-size: 22px; font-weight: bold; padding: 12px; "
            f"background-color: #222; color: {color}; border-radius: 6px;"
        )
        with open(self.status_file, "w") as f:
            f.write("TOOL_DETECTED\n" if detected else "CLEAR\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python viewer.py <frames_dir> <model.pt> [status_file]")
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    model_path = sys.argv[2]
    status_file = sys.argv[3] if len(sys.argv) > 3 else "status.txt"

    app = QApplication(sys.argv)
    win = Viewer(frames_dir, model_path, status_file)
    win.show()
    sys.exit(app.exec())
