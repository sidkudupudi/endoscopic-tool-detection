"""
Watches status.txt (written by the C++ app or the GUI) and forwards state
changes as UART-style messages over a virtual serial port created with:

    socat -d -d pty,raw,echo=0 pty,raw,echo=0

This mirrors exactly how you'd test a serial protocol against a companion
microcontroller before hardware is available.

Usage:
    python serial_writer.py /dev/pts/3 [status_file]
"""
import sys
import time

import serial


def main():
    if len(sys.argv) < 2:
        print("Usage: python serial_writer.py <serial_port> [status_file]")
        sys.exit(1)

    port = sys.argv[1]
    status_file = sys.argv[2] if len(sys.argv) > 2 else "status.txt"

    ser = serial.Serial(port, baudrate=115200, timeout=1)
    last_state = None

    print(f"Watching {status_file}, writing to {port}. Ctrl+C to stop.")
    try:
        while True:
            try:
                with open(status_file) as f:
                    state = f.read().strip()
            except FileNotFoundError:
                time.sleep(0.2)
                continue

            if state != last_state and state:
                msg = f"${state}*\n"  # simple framed protocol: $PAYLOAD*
                ser.write(msg.encode("ascii"))
                print(f"Sent: {msg.strip()}")
                last_state = state

            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
