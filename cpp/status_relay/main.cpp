// Embedded serial relay: watches a status file (written by the Python
// inference/GUI process) and forwards state changes over a real serial
// port using POSIX termios -- no third-party serial library, this talks
// to the device the same way firmware-level embedded C++ does.
//
// Usage: ./status_relay <status_file> <serial_port>
// Example: ./status_relay ../../status.txt /dev/pts/3

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

int openSerialPort(const std::string &path) {
    int fd = open(path.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        std::cerr << "Failed to open " << path << ": " << std::strerror(errno) << "\n";
        return -1;
    }

    termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "tcgetattr failed: " << std::strerror(errno) << "\n";
        close(fd);
        return -1;
    }

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    tty.c_lflag = 0;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_oflag = 0;

    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "tcsetattr failed: " << std::strerror(errno) << "\n";
        close(fd);
        return -1;
    }

    return fd;
}

std::string readStatus(const std::string &statusFile) {
    std::ifstream f(statusFile);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <status_file> <serial_port>\n";
        return 1;
    }

    std::string statusFile = argv[1];
    std::string serialPath = argv[2];

    int fd = openSerialPort(serialPath);
    if (fd < 0) return 1;

    std::cout << "Relaying " << statusFile << " -> " << serialPath
              << " (115200 baud). Ctrl+C to stop.\n";

    std::string lastState;
    while (true) {
        std::string state = readStatus(statusFile);

        if (!state.empty() && state != lastState) {
            std::string msg = "$" + state + "*\n";
            ssize_t written = write(fd, msg.c_str(), msg.size());
            if (written < 0) {
                std::cerr << "Serial write failed: " << std::strerror(errno) << "\n";
            } else {
                std::cout << "Sent: " << msg;
            }
            lastState = state;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    close(fd);
    return 0;
}
