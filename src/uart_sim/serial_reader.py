"""
Simulates the companion microcontroller: reads framed messages ($PAYLOAD*)
off the other end of the virtual serial pair and prints them, as a real
firmware UART receive handler would.

Usage:
    python serial_reader.py /dev/pts/4
"""
import sys

import serial


def main():
    if len(sys.argv) < 2:
        print("Usage: python serial_reader.py <serial_port>")
        sys.exit(1)

    port = sys.argv[1]
    ser = serial.Serial(port, baudrate=115200, timeout=1)

    print(f"Listening on {port}. Ctrl+C to stop.")
    buffer = ""
    try:
        while True:
            byte = ser.read(1)
            if not byte:
                continue
            char = byte.decode("ascii", errors="ignore")
            buffer += char
            if char == "\n":
                buffer = buffer.strip()
                if buffer.startswith("$") and buffer.endswith("*"):
                    payload = buffer[1:-1]
                    print(f"[MCU] Received status: {payload}")
                buffer = ""
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
